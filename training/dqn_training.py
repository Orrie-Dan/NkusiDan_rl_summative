"""DQN training with TorchRL (Trello card 10).

Saves models to models/dqn/<run_id>/ and logs to logs/dqn/<run_id>/ (see ``training/__init__.py``).

Pieces:
- ``QValueActor`` with ``action_mask_key``: greedy action is the argmax over *valid* actions.
- ``EGreedyModule`` with the same mask: exploratory actions are sampled from valid actions only.
- ``DQNLoss`` with a target network (``delay_value=True``) and optional Double DQN. The target Q is computed
  by the same masked ``QValueActor`` on the next observation, so invalid next actions never enter
  ``max_a' Q(s', a')`` (``QValueModule`` overwrites their values with ``finfo.min``). Terminal steps are
  ``terminated`` (shift end), so the bootstrap term is zeroed there.
- ``TensorDictReplayBuffer`` over ``LazyTensorStorage`` (uniform sampling).
- ``SoftUpdate`` (Polyak, ``tau``) or ``HardUpdate`` (copy every ``hard_update_interval`` gradient steps).

Usage::

    uv run python -m training.dqn_training --total-frames 50000 --name dqn_smoke
    uv run python -m training.dqn_training --config path/to/config.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training import (
    RunConfig,
    TrainingRun,
    completed_episode_returns,
    make_env,
    make_qnet,
    parse_run_config,
    seed_everything,
)

METRIC_COLUMNS = ["loss", "q_value_mean", "epsilon", "buffer_size", "grad_norm"]
TARGET_UPDATES = ("soft", "hard")
LOSS_FUNCTIONS = ("l2", "smooth_l1")


@dataclass
class DQNConfig(RunConfig):
    algo: str = "dqn"
    activation: str = "relu"
    lr: float = 5e-4
    frames_per_batch: int = 200
    updates_per_batch: int = 50  # gradient steps per collected batch (0.25 updates per frame by default)
    buffer_size: int = 50_000
    batch_size: int = 128
    learning_starts: int = 5_000  # frames collected before the first gradient step
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frames: int = 60_000  # linear decay from eps_start to eps_end over this many frames
    target_update: str = "soft"
    tau: float = 0.005  # soft update rate
    hard_update_interval: int = 1_000  # gradient steps between hard copies
    double_dqn: bool = False
    loss_function: str = "smooth_l1"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algo != "dqn":
            raise ValueError("DQNConfig.algo must be 'dqn'")
        if self.target_update not in TARGET_UPDATES:
            raise ValueError(f"target_update must be one of {TARGET_UPDATES}")
        if self.loss_function not in LOSS_FUNCTIONS:
            raise ValueError(f"loss_function must be one of {LOSS_FUNCTIONS}")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.batch_size > self.buffer_size or self.eps_decay_frames <= 0 or self.updates_per_batch < 0:
            raise ValueError("need batch_size <= buffer_size, eps_decay_frames > 0, updates_per_batch >= 0")


def train(cfg: DQNConfig, root: str | Path = ".") -> dict:
    """Train one DQN run and return its summary (also written to logs/dqn/<run_id>/summary.json)."""
    seed_everything(cfg.seed)
    env = make_env(cfg.scenario, cfg.seed, cfg.reward_config)
    qactor, q_net = make_qnet(env.action_spec, cfg.hidden_sizes, cfg.activation)
    egreedy = EGreedyModule(
        env.action_spec,
        eps_init=cfg.eps_start,
        eps_end=cfg.eps_end,
        annealing_num_steps=cfg.eps_decay_frames,
        action_mask_key="action_mask",
    )
    policy = TensorDictSequential(qactor, egreedy)

    loss_module = DQNLoss(
        qactor,
        action_space=env.action_spec,
        delay_value=True,
        double_dqn=cfg.double_dqn,
        loss_function=cfg.loss_function,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    if cfg.target_update == "soft":
        target_updater = SoftUpdate(loss_module, tau=cfg.tau)
    else:
        target_updater = HardUpdate(loss_module, value_network_update_interval=cfg.hard_update_interval)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)
    # Same process: the collector runs `policy` itself, which shares parameters with q_net and the loss module,
    # so every gradient step is visible to the next collected batch without a weight sync.
    collector = Collector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        auto_register_policy_transforms=False,
    )
    run = TrainingRun(cfg, metric_columns=METRIC_COLUMNS, root=root)

    frames, updates = 0, 0
    try:
        run.maybe_evaluate(0, q_net)  # untrained reference point
        for batch in collector:
            n = batch.numel()
            frames += n
            buffer.extend(batch.reshape(-1))
            epsilon = float(egreedy.eps)  # epsilon used while collecting this batch
            egreedy.step(n)

            losses, grad_norms, q_means = [], [], []
            if frames >= cfg.learning_starts:
                for _ in range(cfg.updates_per_batch):
                    sample = buffer.sample()
                    loss = loss_module(sample)["loss"]
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    target_updater.step()
                    updates += 1
                    losses.append(loss.detach())
                    grad_norms.append(grad_norm.detach())
                with torch.no_grad():  # mean Q of the taken (valid) actions in the last sampled minibatch
                    q = q_net(sample["observation"]).gather(-1, sample["action"].long().unsqueeze(-1))
                    q_means.append(q.mean())

            run.log(
                frames,
                updates=updates,
                episode_returns=completed_episode_returns(batch),
                loss=torch.stack(losses).mean() if losses else None,
                q_value_mean=q_means[0] if q_means else None,
                epsilon=epsilon,
                buffer_size=len(buffer),
                grad_norm=torch.stack(grad_norms).mean() if grad_norms else None,
            )
            run.maybe_evaluate(frames, q_net)
    finally:
        collector.shutdown()
        env.close()
    return run.finish(q_net, frames, extra={"updates": updates})


# ----------------------------------------------------------------------------------------------------- CLI


def _parse(argv=None) -> tuple[DQNConfig, str]:
    return parse_run_config(DQNConfig, argv, "Train DQN on WasteFlow")


def main(argv=None) -> int:
    cfg, root = _parse(argv)
    summary = train(cfg, root)
    final = summary["final_eval"]
    print(f"{cfg.run_id}: {summary['frames']} frames in {summary['wall_time_s']} s "
          f"({summary['frames_per_s']} fps) | best eval return {summary['best_eval_return']:.2f} "
          f"@ {summary['best_eval_frames']} | final {final['eval_return_mean']:.2f} | {summary['log_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
