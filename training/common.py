"""Shared RL infrastructure for all four algorithms (Trello card 09).

- ``make_env``: TorchRL env (GymEnv + mask cast + ActionMask + RewardSum) for a WasteFlow scenario.
- ``mlp`` / ``make_actor`` / ``make_critic`` / ``make_qnet``: network builders with configurable width/depth.
- ``masked_argmax`` / ``TorchPolicy``: greedy masked action selection from any obs -> scores network
  (actor logits or Q-values), wrapped in the ``evaluation.rollout.Policy`` interface.
- ``RunConfig``: run configuration dataclass <-> JSON. Algorithm modules subclass it with their hyperparameters.
- ``seed_everything``: Python / NumPy / torch seeding and ``torch.set_num_threads(1)``.
- ``TrainingRun``: output paths, CSV training log, periodic greedy evaluation on validation seeds,
  best-checkpoint saving and ``summary.json``.

Output layout (from the brief)::

    models/dqn/<run_id>/            model.pt  model_final.pt  config.json
    models/pg/<algo>_<run_id>/      model.pt  model_final.pt  config.json
    logs/<algo>/<run_id>/           train_log.csv  summary.json  config.json

``model.pt`` is the checkpoint with the best greedy validation return; ``model_final.pt`` is the last one.
A checkpoint stores only the obs -> scores MLP (actor logits for policy-gradient methods, Q-values for DQN),
which is all a greedy policy needs, so ``load_policy`` works the same for every algorithm.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import ActionMask, Compose, DTypeCastTransform, RewardSum
from torchrl.modules import MLP, MaskedCategorical, ProbabilisticActor, QValueActor, ValueOperator
from tensordict.nn import TensorDictModule

from environment.custom_env import ENV_ID, OBS_DIM, RewardConfig, WasteFlowEnv
from environment.simulation import N_ACTIONS
from evaluation.rollout import run_episode

ALGOS = ("dqn", "reinforce", "a2c", "ppo")
PG_ALGOS = ("reinforce", "a2c", "ppo")

# Greedy evaluation during training uses validation seeds, kept apart from training seeds (0, 1, ...) and
# from the final evaluation seeds in evaluation.rollout.EVAL_SEEDS, so model selection never sees test episodes.
VALIDATION_SEEDS: tuple[int, ...] = tuple(range(20_000, 20_010))

ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}

# Columns every train_log.csv has; algorithms add their own metric columns after these.
BASE_LOG_COLUMNS = (
    "frames", "updates", "wall_time_s", "episodes", "train_return_mean",
    "eval_return_mean", "eval_return_std", "eval_tonnes_collected", "eval_overflow_bin_minutes",
    "eval_urgent_service_rate",
)


# ------------------------------------------------------------------------------------------------------ config


@dataclass
class RunConfig:
    """Settings shared by every algorithm. Subclasses add method-specific hyperparameters as fields."""

    algo: str = "dqn"
    name: str = "default"  # config id in the sweep, e.g. "dqn_c03"
    seed: int = 0
    scenario: str = "base"
    total_frames: int = 200_000
    frames_per_batch: int = 1_000
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    lr: float = 3e-4
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    eval_every_frames: int = 10_000
    eval_episodes: int = 10
    reward: dict = field(default_factory=lambda: asdict(RewardConfig()))

    def __post_init__(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"unknown algo {self.algo!r}; expected one of {ALGOS}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}; expected one of {list(ACTIVATIONS)}")
        if self.eval_episodes > len(VALIDATION_SEEDS):
            raise ValueError(f"eval_episodes must be <= {len(VALIDATION_SEEDS)}")

    @property
    def run_id(self) -> str:
        return f"{self.name}_s{self.seed}"

    @property
    def reward_config(self) -> RewardConfig:
        return RewardConfig(**self.reward)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------------------------- env & seeding


def seed_everything(seed: int, threads: int = 1) -> None:
    """Seed Python, NumPy and torch, and pin torch to ``threads`` threads (parallelism is across runs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def make_env(scenario: str = "base", seed: int = 0, reward_config: RewardConfig | None = None) -> TransformedEnv:
    """TorchRL env for training: integer actions, bool ``action_mask``, ``episode_reward`` from RewardSum."""
    env = TransformedEnv(
        GymEnv(ENV_ID, categorical_action_encoding=True, scenario=scenario, reward_config=reward_config),
        Compose(
            DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"]),
            ActionMask(),
            RewardSum(),
        ),
    )
    env.set_seed(seed)
    return env


def completed_episode_returns(batch) -> list[float]:
    """Returns of the episodes that ended inside a collector batch (needs RewardSum)."""
    done = batch["next", "done"].reshape(-1)
    return batch["next", "episode_reward"].reshape(-1)[done].tolist()


# ---------------------------------------------------------------------------------------------------- networks


def mlp(out_features: int, hidden_sizes: list[int], activation: str = "tanh", in_features: int = OBS_DIM) -> MLP:
    """Plain MLP. ``in_features`` is explicit: lazy layers break TorchRL loss construction (card 02)."""
    return MLP(
        in_features=in_features,
        out_features=out_features,
        num_cells=list(hidden_sizes),
        activation_class=ACTIVATIONS[activation],
    )


def make_actor(action_spec, hidden_sizes: list[int], activation: str = "tanh") -> tuple[ProbabilisticActor, MLP]:
    """Masked categorical policy. Returns (actor, logits_net); the logits net is what gets checkpointed."""
    logits_net = mlp(N_ACTIONS, hidden_sizes, activation)
    actor = ProbabilisticActor(
        TensorDictModule(logits_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys={"logits": "logits", "mask": "action_mask"},
        distribution_class=MaskedCategorical,
        spec=action_spec,
        return_log_prob=True,
    )
    return actor, logits_net


def make_critic(hidden_sizes: list[int], activation: str = "tanh") -> ValueOperator:
    return ValueOperator(mlp(1, hidden_sizes, activation), in_keys=["observation"])


def make_qnet(action_spec, hidden_sizes: list[int], activation: str = "relu") -> tuple[QValueActor, MLP]:
    """Masked greedy Q-value actor. Returns (qvalue_actor, q_net); the Q net is what gets checkpointed."""
    q_net = mlp(N_ACTIONS, hidden_sizes, activation)
    actor = QValueActor(q_net, in_keys=["observation"], spec=action_spec, action_mask_key="action_mask")
    return actor, q_net


# --------------------------------------------------------------------------------------------- masked actions


def masked_argmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over the last dim restricted to ``mask`` (bool, same shape). Every row needs one valid action."""
    if not bool(mask.any(-1).all()):
        raise ValueError("every mask row needs at least one valid action")
    return scores.masked_fill(~mask, float("-inf")).argmax(-1)


class TorchPolicy:
    """Greedy masked policy around an obs -> scores network; implements ``evaluation.rollout.Policy``."""

    def __init__(self, net: nn.Module, name: str = "agent") -> None:
        self.net = net
        self.name = name

    def reset(self, seed: int | None = None) -> None:
        pass  # greedy: deterministic

    @torch.no_grad()
    def act(self, obs: dict, env: WasteFlowEnv | None = None) -> int:
        x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
        was_training = self.net.training
        self.net.eval()
        try:
            return int(masked_argmax(self.net(x), mask).item())
        finally:
            self.net.train(was_training)


def greedy_evaluate(net: nn.Module, scenario: str, seeds, reward_config: RewardConfig | None = None) -> list[dict]:
    """KPI rows (see ``evaluation.rollout.run_episode``) for the greedy policy of ``net`` on ``seeds``."""
    env = WasteFlowEnv(scenario, reward_config=reward_config)
    policy = TorchPolicy(net)
    return [run_episode(env, policy, s) for s in seeds]


# --------------------------------------------------------------------------------------------- checkpointing


def model_dir_for(algo: str, run_id: str, root: Path = Path(".")) -> Path:
    if algo == "dqn":
        return root / "models" / "dqn" / run_id
    return root / "models" / "pg" / f"{algo}_{run_id}"


def log_dir_for(algo: str, run_id: str, root: Path = Path(".")) -> Path:
    return root / "logs" / algo / run_id


def save_checkpoint(path: Path, net: nn.Module, config: RunConfig, frames: int, eval_return: float | None) -> None:
    torch.save(
        {
            "state_dict": net.state_dict(),
            "algo": config.algo,
            "hidden_sizes": list(config.hidden_sizes),
            "activation": config.activation,
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "frames": frames,
            "eval_return": eval_return,
        },
        path,
    )


def load_policy(path: str | Path, name: str | None = None) -> TorchPolicy:
    """Greedy policy from a ``model.pt`` written by any of the four trainers."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt["obs_dim"] != OBS_DIM or ckpt["n_actions"] != N_ACTIONS:
        raise ValueError(f"checkpoint was trained for obs_dim={ckpt['obs_dim']}, n_actions={ckpt['n_actions']}")
    net = mlp(N_ACTIONS, ckpt["hidden_sizes"], ckpt["activation"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return TorchPolicy(net, name=name or ckpt["algo"])


# ----------------------------------------------------------------------------------------------- training run


class TrainingRun:
    """Bookkeeping for one training run: dirs, CSV log, greedy evaluation, checkpoints and summary.

    Typical use inside a trainer::

        run = TrainingRun(cfg, metric_columns=["loss", "q_value_mean", "epsilon"])
        for batch in collector:
            ...update...
            run.log(frames, updates=n, episode_returns=completed_episode_returns(batch), loss=..., ...)
            run.maybe_evaluate(frames, q_net)
        run.finish(q_net, frames)
    """

    def __init__(self, config: RunConfig, metric_columns: list[str], root: str | Path = ".") -> None:
        self.config = config
        root = Path(root)
        self.model_dir = model_dir_for(config.algo, config.run_id, root)
        self.log_dir = log_dir_for(config.algo, config.run_id, root)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        config.save(self.model_dir / "config.json")
        config.save(self.log_dir / "config.json")

        clash = set(metric_columns) & set(BASE_LOG_COLUMNS)
        if clash:
            raise ValueError(f"metric columns clash with base columns: {sorted(clash)}")
        self.columns = list(BASE_LOG_COLUMNS) + list(metric_columns)
        self._csv_file = open(self.log_dir / "train_log.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self.columns, restval="")
        self._csv.writeheader()
        self._csv_file.flush()

        self._t0 = time.perf_counter()
        self._episodes = 0
        self._next_eval = 0
        self.best_eval_return = -math.inf
        self.best_eval_frames: int | None = None
        self.eval_history: list[dict] = []
        self._recent_returns: list[float] = []

    @property
    def wall_time(self) -> float:
        return time.perf_counter() - self._t0

    def log(self, frames: int, updates: int = 0, episode_returns: list[float] | None = None, **metrics) -> dict:
        """Write one CSV row. ``metrics`` keys must be among the metric columns given at construction."""
        unknown = set(metrics) - set(self.columns)
        if unknown:
            raise ValueError(f"unknown metric columns: {sorted(unknown)}")
        returns = episode_returns or []
        self._episodes += len(returns)
        self._recent_returns = (self._recent_returns + returns)[-100:]
        row = {
            "frames": frames,
            "updates": updates,
            "wall_time_s": round(self.wall_time, 2),
            "episodes": self._episodes,
            "train_return_mean": _fmt(np.mean(returns)) if returns else "",
        }
        row.update({k: _fmt(v) for k, v in metrics.items()})
        self._csv.writerow(row)
        self._csv_file.flush()
        return row

    def maybe_evaluate(self, frames: int, net: nn.Module, force: bool = False) -> dict | None:
        """Greedy evaluation every ``eval_every_frames`` frames; saves ``model.pt`` when the return improves."""
        if not force and frames < self._next_eval:
            return None
        self._next_eval = frames + self.config.eval_every_frames
        cfg = self.config
        rows = greedy_evaluate(net, cfg.scenario, VALIDATION_SEEDS[: cfg.eval_episodes], cfg.reward_config)
        returns = [r["return"] for r in rows]
        result = {
            "frames": frames,
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_tonnes_collected": float(np.mean([r["tonnes_collected"] for r in rows])),
            "eval_overflow_bin_minutes": float(np.mean([r["overflow_bin_minutes"] for r in rows])),
            "eval_urgent_service_rate": float(np.mean([r["urgent_service_rate"] for r in rows])),
        }
        self.eval_history.append(result)
        self._csv.writerow({"frames": frames, "wall_time_s": round(self.wall_time, 2), "episodes": self._episodes,
                            **{k: _fmt(v) for k, v in result.items() if k != "frames"}})
        self._csv_file.flush()
        if result["eval_return_mean"] > self.best_eval_return:
            self.best_eval_return = result["eval_return_mean"]
            self.best_eval_frames = frames
            save_checkpoint(self.model_dir / "model.pt", net, cfg, frames, self.best_eval_return)
        return result

    def finish(self, net: nn.Module, frames: int, extra: dict | None = None) -> dict:
        """Final evaluation, ``model_final.pt`` and ``summary.json``. Returns the summary."""
        final = self.maybe_evaluate(frames, net, force=True)
        save_checkpoint(self.model_dir / "model_final.pt", net, self.config, frames, final["eval_return_mean"])
        self._csv_file.close()
        summary = {
            "algo": self.config.algo,
            "run_id": self.config.run_id,
            "name": self.config.name,
            "seed": self.config.seed,
            "scenario": self.config.scenario,
            "frames": frames,
            "episodes": self._episodes,
            "wall_time_s": round(self.wall_time, 1),
            "frames_per_s": round(frames / max(self.wall_time, 1e-9), 1),
            "best_eval_return": self.best_eval_return,
            "best_eval_frames": self.best_eval_frames,
            "final_eval": final,
            "train_return_last100_mean": float(np.mean(self._recent_returns)) if self._recent_returns else None,
            "model_dir": self.model_dir.as_posix(),
            "log_dir": self.log_dir.as_posix(),
            "config": self.config.to_dict(),
            **(extra or {}),
        }
        (self.log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def dummy_run(root: str | Path = ".", algo: str = "ppo", total_frames: int = 2_000, seed: int = 0) -> dict:
    """Card 09 smoke run: collect with an untrained masked actor (no updates) through the full output pipeline.

    Exercises the real env, Collector, CSV log, greedy evaluation, checkpoints and summary, so cards 10-13 only
    add the learning part.
    """
    from torchrl.collectors import Collector

    cfg = RunConfig(algo=algo, name="dummy", seed=seed, total_frames=total_frames, frames_per_batch=500,
                    hidden_sizes=[32, 32], eval_every_frames=1_000, eval_episodes=2)
    seed_everything(cfg.seed)
    env = make_env(cfg.scenario, cfg.seed, cfg.reward_config)
    actor, logits_net = make_actor(env.action_spec, cfg.hidden_sizes, cfg.activation)
    run = TrainingRun(cfg, metric_columns=["entropy", "policy_loss", "value_loss"], root=root)
    collector = Collector(env, actor, frames_per_batch=cfg.frames_per_batch, total_frames=cfg.total_frames,
                          auto_register_policy_transforms=False)
    frames = 0
    try:
        for batch in collector:
            frames += batch.numel()
            mask = batch["action_mask"]
            logits = batch["logits"].masked_fill(~mask, float("-inf"))
            entropy = MaskedCategorical(logits=batch["logits"], mask=mask).entropy().mean()
            assert torch.isfinite(logits.max(-1).values).all()
            run.log(frames, episode_returns=completed_episode_returns(batch), entropy=entropy,
                    policy_loss=float("nan"), value_loss=float("nan"))
            run.maybe_evaluate(frames, logits_net)
    finally:
        collector.shutdown()
        env.close()
    return run.finish(logits_net, frames)


def _fmt(v) -> Any:
    if isinstance(v, torch.Tensor):
        v = v.detach().float().mean().item()
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    return v
