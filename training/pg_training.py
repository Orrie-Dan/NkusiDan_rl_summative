"""Policy-gradient training with TorchRL: REINFORCE, A2C and PPO (Trello cards 11-13).

Saves models to models/pg/<algo>_<run_id>/ and logs to logs/<algo>/<run_id>/ (see ``training.common``).

REINFORCE (card 11). TorchRL has no REINFORCE loss, so it is written here by hand:

- ``ProbabilisticActor`` + ``MaskedCategorical`` (``training.common.make_actor``): actions are sampled from
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
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical

from training.common import (
    RunConfig,
    TrainingRun,
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


# ----------------------------------------------------------------------------------------------------- CLI

TRAINERS = {"reinforce": (ReinforceConfig, train_reinforce)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in TRAINERS:
        print(f"usage: python -m training.pg_training {{{','.join(TRAINERS)}}} [--flags]  (A2C/PPO: cards 12-13)")
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
