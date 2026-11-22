"""Policy-gradient training with TorchRL: REINFORCE, A2C and PPO (Trello cards 11-13).

Saves models to models/pg/<algo>_<run_id>/ and logs to logs/<algo>/<run_id>/ (see ``training/__init__.py``).

REINFORCE (card 11). TorchRL has no REINFORCE loss, so it is written here by hand:

- ``ProbabilisticActor`` + ``MaskedCategorical`` (``training.make_actor``): actions are sampled from
  valid actions only, and log-probabilities / entropy are over valid actions only.
- Each update collects ``episodes_per_update`` complete episodes (shifts end by termination, never truncation).
- Monte-Carlo reward-to-go ``G_t = sum_k gamma^k r_{t+k}``, computed per episode.
- Baseline: ``none``, ``moving_average`` (EMA of the mean reward-to-go of past batches, a scalar) or ``value``
  (a learned V(s) fitted to ``G_t`` by MSE with its own optimizer; it never shares parameters with the policy).
  Advantage ``A_t = G_t - b``; optionally normalised to zero mean, unit std within the batch.
- Loss ``-mean(log pi(a_t|s_t) * A_t) - entropy_coef * mean(H(pi(.|s_t)))``, one gradient step per batch.

``frames_per_batch`` (from ``RunConfig``) is not used by REINFORCE: batches are whole episodes, so batch size is
``episodes_per_update`` x episode length (about 50-65 decisions per shift in the base scenario).

Usage::

    uv run python -m training.pg_training reinforce --total-frames 100000 --name reinforce_smoke
    uv run python -m training.pg_training reinforce --config path/to/config.json
    uv run python -m training.pg_training a2c --total-frames 100000 --name a2c_smoke
    uv run python -m training.pg_training ppo --total-frames 100000 --name ppo_smoke

A2C (card 12): TorchRL ``A2CLoss`` + ``GAE`` on ``Collector`` batches of ``frames_per_batch`` decisions, one
gradient step per batch, separate actor and critic MLPs, one Adam optimizer, gradient-norm clipping
(see ``train_a2c``).

PPO (card 13): TorchRL ``ClipPPOLoss`` + ``GAE``; each rollout of ``frames_per_batch`` decisions is reused for
``num_epochs`` passes of shuffled minibatches through a replay buffer, with optional linear lr annealing
(see ``train_ppo``).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from training import (
    RunConfig,
    TrainingRun,
    completed_episode_returns,
    make_actor,
    make_critic,
    make_env,
    parse_run_config,
    seed_everything,
)

REINFORCE_METRIC_COLUMNS = ["policy_loss", "entropy", "value_loss", "baseline_mean", "advantage_std", "grad_norm"]
BASELINES = ("none", "moving_average", "value")
MAX_EPISODE_STEPS = 10_000  # safety cap only; a shift ends after ~50-65 decisions


@dataclass
class ReinforceConfig(RunConfig):
    algo: str = "reinforce"
    lr: float = 3e-3  # 1e-2 collapsed entropy to ~0.1 in a 100k-frame test; 1e-3 barely moved it
    episodes_per_update: int = 5
    baseline: str = "value"
    baseline_momentum: float = 0.9  # moving_average: b <- m * b + (1 - m) * mean(G) after each batch
    value_lr: float = 1e-3  # value baseline: Adam learning rate
    value_steps_per_update: int = 5  # value baseline: full-batch MSE steps per update
    normalize_advantages: bool = True
    entropy_coef: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algo != "reinforce":
            raise ValueError("ReinforceConfig.algo must be 'reinforce'")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        if not 0.0 <= self.baseline_momentum < 1.0:
            raise ValueError("baseline_momentum must be in [0, 1)")
        if self.episodes_per_update < 1 or self.value_steps_per_update < 1:
            raise ValueError("need episodes_per_update >= 1 and value_steps_per_update >= 1")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be >= 0")


def reward_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted reward-to-go of ONE episode: ``G_t = r_t + gamma * G_{t+1}``, ``G_T = r_T`` (terminal)."""
    rewards = rewards.reshape(-1)
    out = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype)
    for t in range(rewards.numel() - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def reinforce_loss(logits_net: torch.nn.Module, batch: TensorDict, advantages: torch.Tensor,
                   entropy_coef: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total_loss, policy_loss, entropy). Log-probs are recomputed with grad from the stored observations."""
    dist = MaskedCategorical(logits=logits_net(batch["observation"]), mask=batch["action_mask"])
    log_prob = dist.log_prob(batch["action"])
    policy_loss = -(log_prob * advantages).mean()
    entropy = dist.entropy().mean()
    return policy_loss - entropy_coef * entropy, policy_loss, entropy


def collect_episodes(env, actor, n_episodes: int) -> list[TensorDict]:
    """``n_episodes`` complete episodes sampled from the stochastic (masked) policy."""
    episodes = []
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        for _ in range(n_episodes):
            td = env.rollout(MAX_EPISODE_STEPS, actor, break_when_any_done=True)
            if not bool(td["next", "done"][-1]):
                raise RuntimeError(f"episode did not finish within {MAX_EPISODE_STEPS} steps")
            episodes.append(td)
    return episodes


def train_reinforce(cfg: ReinforceConfig, root: str | Path = ".") -> dict:
    """Train one REINFORCE run and return its summary (also written to logs/reinforce/<run_id>/summary.json)."""
    seed_everything(cfg.seed)
    env = make_env(cfg.scenario, cfg.seed, cfg.reward_config)
    actor, logits_net = make_actor(env.action_spec, cfg.hidden_sizes, cfg.activation)
    optimizer = torch.optim.Adam(logits_net.parameters(), lr=cfg.lr)
    value_net = value_optimizer = None
    if cfg.baseline == "value":
        value_net = make_critic(cfg.hidden_sizes, cfg.activation).module  # plain obs -> V(s) MLP
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)
    moving_baseline: float | None = None

    run = TrainingRun(cfg, metric_columns=REINFORCE_METRIC_COLUMNS, root=root)
    frames, updates = 0, 0
    try:
        run.maybe_evaluate(0, logits_net)  # untrained reference point
        while frames < cfg.total_frames:
            episodes = collect_episodes(env, actor, cfg.episodes_per_update)
            batch = torch.cat(episodes)
            returns = torch.cat([reward_to_go(ep["next", "reward"], cfg.gamma) for ep in episodes])
            frames += batch.numel()

            value_loss = None
            if cfg.baseline == "value":
                with torch.no_grad():
                    baseline = value_net(batch["observation"]).reshape(-1)
                for _ in range(cfg.value_steps_per_update):
                    v = value_net(batch["observation"]).reshape(-1)
                    value_loss = torch.nn.functional.mse_loss(v, returns)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), cfg.max_grad_norm)
                    value_optimizer.step()
            elif cfg.baseline == "moving_average":
                batch_mean = float(returns.mean())
                b = batch_mean if moving_baseline is None else moving_baseline
                baseline = torch.full_like(returns, b)
                moving_baseline = cfg.baseline_momentum * b + (1 - cfg.baseline_momentum) * batch_mean
            else:
                baseline = torch.zeros_like(returns)

            advantages = returns - baseline
            if cfg.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            loss, policy_loss, entropy = reinforce_loss(logits_net, batch, advantages, cfg.entropy_coef)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(logits_net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            updates += 1

            run.log(
                frames,
                updates=updates,
                episode_returns=[float(ep["next", "episode_reward"][-1]) for ep in episodes],
                policy_loss=policy_loss,
                entropy=entropy,
                value_loss=value_loss,
                baseline_mean=baseline.mean(),
                advantage_std=advantages.std() if advantages.numel() > 1 else None,
                grad_norm=grad_norm,
            )
            run.maybe_evaluate(frames, logits_net)
    finally:
        env.close()
    return run.finish(logits_net, frames, extra={"updates": updates})


# ----------------------------------------------------------------------------------------------------- A2C

A2C_METRIC_COLUMNS = ["policy_loss", "entropy", "value_loss", "advantage_std", "explained_variance", "grad_norm"]
CRITIC_LOSSES = ("l2", "smooth_l1")


@dataclass
class A2CConfig(RunConfig):
    algo: str = "a2c"
    lr: float = 1e-3
    frames_per_batch: int = 256  # rollout length: decisions collected per update (spans episode boundaries)
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    loss_critic_type: str = "smooth_l1"
    normalize_advantages: bool = True
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algo != "a2c":
            raise ValueError("A2CConfig.algo must be 'a2c'")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.entropy_coef < 0 or self.value_coef <= 0:
            raise ValueError("need entropy_coef >= 0 and value_coef > 0")
        if self.loss_critic_type not in CRITIC_LOSSES:
            raise ValueError(f"loss_critic_type must be one of {CRITIC_LOSSES}")
        if self.frames_per_batch < 2 or self.max_grad_norm <= 0:
            raise ValueError("need frames_per_batch >= 2 and max_grad_norm > 0")


def explained_variance(values: torch.Tensor, targets: torch.Tensor) -> float:
    """``1 - Var(target - V) / Var(target)``: 1 is a perfect critic, 0 no better than a constant, < 0 worse."""
    var = targets.var()
    return float("nan") if var <= 0 else float(1 - (targets - values).var() / var)


def train_a2c(cfg: A2CConfig, root: str | Path = ".") -> dict:
    """Train one A2C run and return its summary (also written to logs/a2c/<run_id>/summary.json).

    Synchronous advantage actor-critic on one env: collect ``frames_per_batch`` decisions with the masked policy,
    compute GAE(gamma, lambda) advantages and value targets with the current critic (bootstrapping at the batch
    end, zeroed at shift end), then ONE gradient step on
    ``loss_objective + value_coef * loss_critic - entropy_coef * entropy`` (TorchRL ``A2CLoss``). Actor and
    critic are separate MLPs updated by one Adam optimizer, with global gradient-norm clipping.
    """
    seed_everything(cfg.seed)
    env = make_env(cfg.scenario, cfg.seed, cfg.reward_config)
    actor, logits_net = make_actor(env.action_spec, cfg.hidden_sizes, cfg.activation)
    critic = make_critic(cfg.hidden_sizes, cfg.activation)
    advantage = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic,
                    average_gae=cfg.normalize_advantages)
    loss_module = A2CLoss(actor, critic, entropy_coeff=cfg.entropy_coef, critic_coeff=cfg.value_coef,
                          loss_critic_type=cfg.loss_critic_type)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    # Same process and shared parameters (checked in the tests): each update is visible to the next batch.
    collector = Collector(env, actor, frames_per_batch=cfg.frames_per_batch, total_frames=cfg.total_frames,
                          auto_register_policy_transforms=False)

    run = TrainingRun(cfg, metric_columns=A2C_METRIC_COLUMNS, root=root)
    frames, updates = 0, 0
    try:
        run.maybe_evaluate(0, logits_net)  # untrained reference point
        for batch in collector:
            frames += batch.numel()
            with torch.no_grad():
                advantage(batch)  # writes "advantage", "value_target" and "state_value"
            out = loss_module(batch)
            loss = sum(v for k, v in out.items() if k.startswith("loss_"))  # no loss_entropy if entropy_coef == 0
            if "entropy" in out.keys():
                entropy = out["entropy"]
            else:  # batch logits come from the current parameters: one update per batch, collected just before
                entropy = MaskedCategorical(logits=batch["logits"], mask=batch["action_mask"]).entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
            optimizer.step()
            updates += 1

            run.log(
                frames,
                updates=updates,
                episode_returns=completed_episode_returns(batch),
                policy_loss=out["loss_objective"],
                entropy=entropy,
                value_loss=out["loss_critic"] / cfg.value_coef,  # unweighted critic loss
                advantage_std=batch["advantage"].std(),
                explained_variance=explained_variance(batch["state_value"], batch["value_target"]),
                grad_norm=grad_norm,
            )
            run.maybe_evaluate(frames, logits_net)
    finally:
        collector.shutdown()
        env.close()
    return run.finish(logits_net, frames, extra={"updates": updates})


# ----------------------------------------------------------------------------------------------------- PPO

PPO_METRIC_COLUMNS = ["policy_loss", "entropy", "value_loss", "kl_approx", "clip_fraction", "advantage_std",
                      "explained_variance", "grad_norm", "lr"]


@dataclass
class PPOConfig(RunConfig):
    algo: str = "ppo"
    lr: float = 3e-4
    frames_per_batch: int = 1_024  # rollout: decisions collected per PPO iteration (~18 shifts)
    num_epochs: int = 4  # passes over each rollout
    minibatch_size: int = 128
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    loss_critic_type: str = "smooth_l1"
    normalize_advantages: bool = True  # over the whole rollout (GAE ``average_gae``), as in A2C
    anneal_lr: bool = False  # linear decay from lr to 0 over the run, stepped once per rollout
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algo != "ppo":
            raise ValueError("PPOConfig.algo must be 'ppo'")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.entropy_coef < 0 or self.value_coef <= 0:
            raise ValueError("need entropy_coef >= 0 and value_coef > 0")
        if self.loss_critic_type not in CRITIC_LOSSES:
            raise ValueError(f"loss_critic_type must be one of {CRITIC_LOSSES}")
        if self.num_epochs < 1 or not 1 <= self.minibatch_size <= self.frames_per_batch:
            raise ValueError("need num_epochs >= 1 and 1 <= minibatch_size <= frames_per_batch")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")

    @property
    def minibatches_per_epoch(self) -> int:
        return self.frames_per_batch // self.minibatch_size  # a remainder of the shuffled rollout is skipped


def train_ppo(cfg: PPOConfig, root: str | Path = ".") -> dict:
    """Train one PPO run and return its summary (also written to logs/ppo/<run_id>/summary.json).

    Per rollout of ``frames_per_batch`` decisions: GAE(gamma, lambda) advantages and value targets from the critic
    as it was at collection time, then ``num_epochs`` passes over the rollout in shuffled minibatches
    (``SamplerWithoutReplacement``) of the clipped surrogate ``ClipPPOLoss`` (+ value_coef * critic loss
    - entropy_coef * entropy), one clipped Adam step per minibatch. ``action_log_prob`` stored by the collector is
    the behaviour policy for the importance ratio.
    """
    seed_everything(cfg.seed)
    env = make_env(cfg.scenario, cfg.seed, cfg.reward_config)
    actor, logits_net = make_actor(env.action_spec, cfg.hidden_sizes, cfg.activation)
    critic = make_critic(cfg.hidden_sizes, cfg.activation)
    advantage = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic,
                    average_gae=cfg.normalize_advantages)
    loss_module = ClipPPOLoss(actor, critic, clip_epsilon=cfg.clip_epsilon, entropy_coeff=cfg.entropy_coef,
                              critic_coeff=cfg.value_coef, loss_critic_type=cfg.loss_critic_type)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    n_rollouts = math.ceil(cfg.total_frames / cfg.frames_per_batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: max(0.0, 1.0 - i / n_rollouts) if cfg.anneal_lr else 1.0)
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.frames_per_batch),
                                    sampler=SamplerWithoutReplacement(), batch_size=cfg.minibatch_size)
    collector = Collector(env, actor, frames_per_batch=cfg.frames_per_batch, total_frames=cfg.total_frames,
                          auto_register_policy_transforms=False)

    run = TrainingRun(cfg, metric_columns=PPO_METRIC_COLUMNS, root=root)
    frames, updates = 0, 0
    try:
        run.maybe_evaluate(0, logits_net)  # untrained reference point
        for batch in collector:
            frames += batch.numel()
            lr = optimizer.param_groups[0]["lr"]  # lr used for this rollout's updates
            with torch.no_grad():
                advantage(batch)  # writes "advantage", "value_target" and "state_value"
            buffer.empty()
            buffer.extend(batch.reshape(-1))

            stats = {k: [] for k in ("loss_objective", "entropy", "loss_critic", "kl_approx", "clip_fraction")}
            grad_norms = []
            for _ in range(cfg.num_epochs):
                for _ in range(cfg.minibatches_per_epoch):
                    out = loss_module(buffer.sample())
                    loss = sum(v for k, v in out.items() if k.startswith("loss_"))
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norms.append(torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm))
                    optimizer.step()
                    updates += 1
                    for k in stats:
                        stats[k].append(out[k].detach())
            scheduler.step()

            mean = {k: torch.stack(v).mean() for k, v in stats.items()}
            run.log(
                frames,
                updates=updates,
                episode_returns=completed_episode_returns(batch),
                policy_loss=mean["loss_objective"],
                entropy=mean["entropy"],
                value_loss=mean["loss_critic"] / cfg.value_coef,  # unweighted critic loss
                kl_approx=mean["kl_approx"],
                clip_fraction=mean["clip_fraction"],
                advantage_std=batch["advantage"].std(),
                explained_variance=explained_variance(batch["state_value"], batch["value_target"]),
                grad_norm=torch.stack(grad_norms).mean(),
                lr=lr,
            )
            run.maybe_evaluate(frames, logits_net)
    finally:
        collector.shutdown()
        env.close()
    return run.finish(logits_net, frames, extra={"updates": updates})


# ----------------------------------------------------------------------------------------------------- CLI

TRAINERS = {"reinforce": (ReinforceConfig, train_reinforce), "a2c": (A2CConfig, train_a2c),
            "ppo": (PPOConfig, train_ppo)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in TRAINERS:
        print(f"usage: python -m training.pg_training {{{','.join(TRAINERS)}}} [--flags]")
        return 2
    algo, flags = argv[0], argv[1:]
    config_cls, trainer = TRAINERS[algo]
    cfg, root = parse_run_config(config_cls, flags, f"Train {algo} on WasteFlow")
    summary = trainer(cfg, root)
    final = summary["final_eval"]
    print(f"{cfg.run_id}: {summary['frames']} frames in {summary['wall_time_s']} s "
          f"({summary['frames_per_s']} fps) | best eval return {summary['best_eval_return']:.2f} "
          f"@ {summary['best_eval_frames']} | final {final['eval_return_mean']:.2f} | {summary['log_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
