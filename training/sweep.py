"""Hyperparameter sweep: 40 configs (10 per algorithm) and a parallel, resumable runner (Trello card 15).

Configs (``SWEEP``): for each algorithm, config ``c00`` is the base config (the trainer dataclass defaults,
tuned in cards 10-13) and ``c01``-``c09`` each change ONE factor from it (one-factor-at-a-time). Every entry
carries the factor it varies and a short rationale for the report. All runs use the same frame budget
(``RunConfig.total_frames``, 200k since env v1.0).

Runner (``run_jobs``): every run is its own subprocess (``python -m training.<trainer> --config <json>``), so a
crash, even a hard one, takes down only that run. A thread pool keeps ``workers`` subprocesses going.
Resumable: a run whose ``summary.json`` exists with the same config is skipped. A summary with a DIFFERENT
config is reported as ``stale`` and left alone unless ``overwrite=True``. Each run's stdout/stderr goes to
``<root>/logs/sweep/runs/<run_id>.log``. When the sweep ends, ``<root>/logs/sweep/<tag>_results.csv`` collects
one row per run from the summaries, and ``<tag>_status.json`` records how every job ended.

Usage::

    uv run main.py sweep --list                    # print the 40 configs
    uv run main.py sweep                           # Phase 1: 40 runs, seed 0, into logs/ and models/
    uv run main.py sweep --dry-run                 # tiny budgets into logs/_scratch/sweep_dry/
    uv run main.py sweep --configs dqn_c08,ppo_c04 --seeds 0,1,2,3,4 --total-frames 600000 --name-suffix _p2 --tag phase2  # Phase 2: picked configs x 5 seeds, own run names
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.common import ALGOS, RunConfig, log_dir_for
from training.dqn_training import DQNConfig
from training.pg_training import A2CConfig, PPOConfig, ReinforceConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_CLASSES: dict[str, type[RunConfig]] = {
    "dqn": DQNConfig, "reinforce": ReinforceConfig, "a2c": A2CConfig, "ppo": PPOConfig,
}
SWEEP_DIR = Path("logs") / "sweep"

# Dry run: tiny budgets that still reach every code path (DQN gets past learning_starts, PPO/A2C do updates).
DRY_RUN_FRAMES = 4_096
DRY_RUN_OVERRIDES = {"eval_every_frames": 2_048, "eval_episodes": 2}


@dataclass(frozen=True)
class SweepEntry:
    """One sweep config: ``overrides`` applied to the algorithm's base config."""

    algo: str
    index: int
    factor: str  # what this config varies ("base" for c00)
    rationale: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.algo}_c{self.index:02d}"


def _entries(algo: str, rows: list[tuple[str, dict, str]]) -> list[SweepEntry]:
    base = SweepEntry(algo, 0, "base", "Base config: trainer defaults tuned in the single-algorithm cards.")
    return [base] + [SweepEntry(algo, i, factor, why, ov) for i, (factor, ov, why) in enumerate(rows, start=1)]


# Base values (for reading the table): see DQNConfig / ReinforceConfig / A2CConfig / PPOConfig.
SWEEP: dict[str, list[SweepEntry]] = {
    # base: lr 5e-4, gamma 0.99, eps 1.0 -> 0.05 over 60k frames, buffer 50k, batch 128, learning_starts 5k,
    #       soft target tau 0.005, 0.25 updates/frame, [128,128] relu
    "dqn": _entries("dqn", [
        ("lr", {"lr": 1e-4}, "Lower learning rate: slower but steadier Q-value fitting."),
        ("lr", {"lr": 1e-3}, "Higher learning rate: faster progress, risk of unstable bootstrapped targets."),
        ("gamma", {"gamma": 0.95}, "Shorter horizon (~20 decisions): favours immediate pickups over planning."),
        ("exploration", {"eps_decay_frames": 20_000}, "Fast epsilon decay: exploit early, risk locking in a poor route habit."),
        ("exploration", {"eps_decay_frames": 120_000}, "Slow epsilon decay: explore for 60% of the budget."),
        ("replay", {"buffer_size": 10_000}, "Small replay buffer: fresher, more on-policy data, less diversity."),
        ("replay", {"batch_size": 32}, "Small minibatch: noisier gradients, same number of updates."),
        ("replay", {"learning_starts": 20_000}, "Late learning start: a more diverse buffer before the first update."),
        ("target_update", {"target_update": "hard"},
         "Hard target copy every 1k gradient steps (hard_update_interval) instead of soft Polyak updates."),
    ]),
    # base: lr 3e-3, gamma 0.99, value baseline, normalised advantages, entropy 0.01, 5 episodes per update
    "reinforce": _entries("reinforce", [
        ("lr", {"lr": 1e-3}, "Lower learning rate: smaller steps on high-variance Monte-Carlo gradients."),
        ("lr", {"lr": 1e-2}, "Higher learning rate: tests whether the policy collapses early (low entropy)."),
        ("gamma", {"gamma": 0.95}, "Shorter horizon: lower-variance returns, less credit for long-term effects."),
        ("baseline", {"baseline": "none"}, "No baseline: plain REINFORCE, the variance-reduction reference."),
        ("baseline", {"baseline": "moving_average"}, "Scalar moving-average baseline instead of a learned V(s)."),
        ("normalization", {"normalize_advantages": False}, "Raw advantages: no per-batch scale normalisation."),
        ("entropy", {"entropy_coef": 0.05}, "Stronger entropy bonus: keeps exploring longer."),
        ("episodes_per_update", {"episodes_per_update": 2}, "Fewer episodes per update: more, noisier updates."),
        ("episodes_per_update", {"episodes_per_update": 10}, "More episodes per update: fewer, lower-variance updates."),
    ]),
    # base: lr 1e-3, gamma 0.99, rollout 256, GAE lambda 0.95, entropy 0.01, value coef 0.5
    "a2c": _entries("a2c", [
        ("lr", {"lr": 3e-4}, "Lower learning rate: steadier but slower actor-critic updates."),
        ("lr", {"lr": 3e-3}, "Higher learning rate: faster, risk of policy collapse with one step per batch."),
        ("rollout_length", {"frames_per_batch": 64}, "Short rollouts (~1 shift): many small, noisy updates."),
        ("rollout_length", {"frames_per_batch": 1_024}, "Long rollouts (~18 shifts): few, low-variance updates."),
        ("entropy", {"entropy_coef": 0.05}, "Stronger entropy bonus: more exploration, slower commitment."),
        ("value_coef", {"value_coef": 0.25}, "Lighter critic loss weight relative to the policy loss."),
        ("value_coef", {"value_coef": 1.0}, "Heavier critic loss weight: faster value fitting."),
        ("gae_lambda", {"gae_lambda": 0.8}, "Lower GAE lambda: leans on the critic, lower variance, more bias."),
        ("gae_lambda", {"gae_lambda": 1.0}, "GAE lambda 1: Monte-Carlo-like advantages, high variance, no bias."),
    ]),
    # base: lr 3e-4, gamma 0.99, rollout 1024, 4 epochs, minibatch 128, clip 0.2, GAE 0.95, entropy 0.01, vf 0.5
    "ppo": _entries("ppo", [
        ("lr", {"lr": 1e-3}, "Higher learning rate: faster progress per rollout, tests the clip's protection."),
        ("clip_epsilon", {"clip_epsilon": 0.1}, "Tighter clip: more conservative policy updates."),
        ("clip_epsilon", {"clip_epsilon": 0.3}, "Looser clip: larger policy changes per rollout."),
        ("epochs", {"num_epochs": 10}, "More epochs per rollout: better sample use, risk of overfitting a rollout."),
        ("minibatch_size", {"minibatch_size": 256}, "Larger minibatch: half the gradient steps, smoother each."),
        ("rollout_length", {"frames_per_batch": 2_048}, "Longer rollout (~36 shifts): fewer, better-averaged updates."),
        ("entropy", {"entropy_coef": 0.05}, "Stronger entropy bonus: more exploration."),
        ("value_coef", {"value_coef": 1.0}, "Heavier critic loss weight."),
        ("gae_lambda", {"gae_lambda": 0.9}, "Lower GAE lambda: lower-variance, more biased advantages."),
    ]),
}

ALL_ENTRIES: dict[str, SweepEntry] = {e.name: e for entries in SWEEP.values() for e in entries}


def build_config(entry: SweepEntry, seed: int = 0, total_frames: int | None = None,
                 extra: dict[str, Any] | None = None, name_suffix: str = "") -> RunConfig:
    """The full run config for ``entry``: base defaults + entry overrides + ``total_frames`` / ``extra``.

    ``name_suffix`` (e.g. ``_p2``) gives the run its own name, so runs of the same entry with another budget get
    their own folders instead of colliding with (and being reported stale against) earlier runs.
    """
    kw: dict[str, Any] = {"name": entry.name + name_suffix, "seed": seed, **entry.overrides}
    if total_frames is not None:
        kw["total_frames"] = total_frames
    kw.update(extra or {})
    return CONFIG_CLASSES[entry.algo](**kw)


def dry_run_config(entry: SweepEntry, seed: int = 0) -> RunConfig:
    """Tiny-budget version of ``entry`` for checking the pipeline (DQN starts learning within the budget)."""
    extra = dict(DRY_RUN_OVERRIDES)
    if entry.algo == "dqn":
        extra["learning_starts"] = min(entry.overrides.get("learning_starts", 5_000), 1_000)
    return build_config(entry, seed, DRY_RUN_FRAMES, extra)


def select_entries(algos: list[str] | None = None, configs: list[str] | None = None) -> list[SweepEntry]:
    """Entries filtered by algorithm and config, in table order.

    A config is an index (``c03`` or ``3``, applied to every selected algorithm) or a full name (``dqn_c08``).
    """
    algos = algos or list(ALGOS)
    unknown = set(algos) - set(ALGOS)
    if unknown:
        raise ValueError(f"unknown algos {sorted(unknown)}")
    if not configs:
        return [e for a in algos for e in SWEEP[a]]
    names = {c for c in configs if c in ALL_ENTRIES}
    bad = [c for c in configs if c not in names and not c.lstrip("c").isdigit()]
    if bad:
        raise ValueError(f"unknown configs {bad}")
    indices = {int(c.lstrip("c")) for c in configs if c not in names}
    return [e for a in algos for e in SWEEP[a] if e.name in names or e.index in indices]


# ----------------------------------------------------------------------------------------------------- runner


def _child_path(path: Path) -> Path:
    """``path`` as the child process (cwd = project root) should see it: relative when inside the project, so
    summary.json files committed under logs/ hold no machine-specific absolute paths."""
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _command(cfg: RunConfig, config_path: Path, root: Path) -> list[str]:
    if cfg.algo == "dqn":
        return [sys.executable, "-m", "training.dqn_training", "--config", str(config_path), "--root", str(root)]
    return [sys.executable, "-m", "training.pg_training", cfg.algo, "--config", str(config_path), "--root", str(root)]


def job_state(cfg: RunConfig, root: Path) -> str:
    """``todo`` (no summary), ``done`` (summary with this exact config) or ``stale`` (summary, other config)."""
    summary_path = log_dir_for(cfg.algo, cfg.run_id, root) / "summary.json"
    if not summary_path.exists():
        return "todo"
    try:
        saved = json.loads(summary_path.read_text(encoding="utf-8")).get("config")
    except (OSError, json.JSONDecodeError):
        return "todo"  # unreadable summary: the run never finished cleanly
    return "done" if saved == cfg.to_dict() else "stale"


def run_job(cfg: RunConfig, root: Path, timeout_s: float | None = None) -> dict[str, Any]:
    """Run one config in its own subprocess. Never raises: failures come back as ``status="failed"``."""
    sweep_dir = root / SWEEP_DIR
    (sweep_dir / "configs").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)
    config_path = sweep_dir / "configs" / f"{cfg.algo}_{cfg.run_id}.json"
    log_path = sweep_dir / "runs" / f"{cfg.algo}_{cfg.run_id}.log"
    cfg.save(config_path)
    env = {**os.environ, "RL_WARNINGS": "False", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONUNBUFFERED": "1"}
    t0 = time.perf_counter()
    result = {"algo": cfg.algo, "run_id": cfg.run_id, "log": log_path.as_posix()}
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(_command(cfg, _child_path(config_path), _child_path(root)), cwd=PROJECT_ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout_s)
        ok = proc.returncode == 0 and job_state(cfg, root) == "done"
        result.update(status="done" if ok else "failed", returncode=proc.returncode)
    except subprocess.TimeoutExpired:
        result.update(status="failed", returncode=None, error=f"timeout after {timeout_s} s")
    except Exception as exc:  # noqa: BLE001 - isolation: one bad job must not stop the sweep
        result.update(status="failed", returncode=None, error=repr(exc))
    result["wall_time_s"] = round(time.perf_counter() - t0, 1)
    return result


def run_jobs(configs: list[RunConfig], root: str | Path = ".", workers: int = 4, overwrite: bool = False,
             tag: str = "sweep", timeout_s: float | None = None, echo=print) -> list[dict[str, Any]]:
    """Run ``configs`` with ``workers`` parallel subprocesses; skip finished runs. Returns one status per config."""
    root = Path(root)
    run_ids = [(c.algo, c.run_id) for c in configs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("duplicate (algo, run_id) in the job list")

    statuses: list[dict[str, Any]] = []
    todo: list[RunConfig] = []
    for cfg in configs:
        state = job_state(cfg, root)
        if state == "done" or (state == "stale" and not overwrite):
            statuses.append({"algo": cfg.algo, "run_id": cfg.run_id, "status": "skipped" if state == "done" else "stale"})
            if state == "stale":
                echo(f"  stale  {cfg.algo}/{cfg.run_id}: summary.json has a different config (use --overwrite)")
        else:
            todo.append(cfg)
    echo(f"{len(configs)} jobs: {len(todo)} to run, {len(configs) - len(todo)} skipped, {workers} workers")

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(run_job, cfg, root, timeout_s): cfg for cfg in todo}
            for i, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                statuses.append(res)
                echo(f"  [{i}/{len(todo)}] {res['status']:6s} {res['algo']}/{res['run_id']} "
                     f"({res['wall_time_s']} s){'' if res['status'] == 'done' else '  log: ' + res['log']}")

    sweep_dir = root / SWEEP_DIR
    sweep_dir.mkdir(parents=True, exist_ok=True)
    order = {rid: i for i, rid in enumerate(run_ids)}
    statuses.sort(key=lambda s: order[(s["algo"], s["run_id"])])
    (sweep_dir / f"{tag}_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    write_results(configs, root, sweep_dir / f"{tag}_results.csv")
    return statuses


RESULT_COLUMNS = (
    "algo", "name", "seed", "factor", "change", "best_eval_return", "best_eval_frames", "final_eval_return",
    "final_tonnes_collected", "final_overflow_bin_minutes", "final_urgent_service_rate", "final_greedy_return",
    "train_return_last100_mean", "frames", "episodes", "wall_time_s", "frames_per_s",
)


def entry_for(run_name: str) -> SweepEntry | None:
    """The sweep entry a run name comes from (``ppo_c04`` or a suffixed ``ppo_c04_p2``), if any."""
    i = run_name.find("_c")  # entry names are <algo>_cNN
    return ALL_ENTRIES.get(run_name[: i + 4]) if i >= 0 else None


def write_results(configs: list[RunConfig], root: Path, path: Path) -> int:
    """One CSV row per finished run (read from its summary.json). Returns the number of rows written."""
    rows = []
    for cfg in configs:
        summary_path = log_dir_for(cfg.algo, cfg.run_id, root) / "summary.json"
        if job_state(cfg, root) != "done":
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        entry = entry_for(cfg.name)
        final = s["final_eval"]
        rows.append({
            "algo": cfg.algo, "name": cfg.name, "seed": cfg.seed,
            "factor": entry.factor if entry else "",
            "change": json.dumps(entry.overrides) if entry and entry.overrides else "",
            "best_eval_return": s["best_eval_return"], "best_eval_frames": s["best_eval_frames"],
            "final_eval_return": final["eval_return_mean"],
            "final_tonnes_collected": final["eval_tonnes_collected"],
            "final_overflow_bin_minutes": final["eval_overflow_bin_minutes"],
            "final_urgent_service_rate": final["eval_urgent_service_rate"],
            "final_greedy_return": final["eval_greedy_return_mean"],
            "train_return_last100_mean": s["train_return_last100_mean"],
            "frames": s["frames"], "episodes": s["episodes"], "wall_time_s": s["wall_time_s"],
            "frames_per_s": s["frames_per_s"],
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# -------------------------------------------------------------------------------------------------------- CLI


def format_table(entries: list[SweepEntry]) -> str:
    lines = [f"{'name':15s} {'factor':20s} {'change':45s} rationale"]
    for e in entries:
        change = ", ".join(f"{k}={v}" for k, v in e.overrides.items()) or "-"
        lines.append(f"{e.name:15s} {e.factor:20s} {change:45s} {e.rationale}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="WasteFlow hyperparameter sweep (40 configs, card 15)")
    p.add_argument("--list", action="store_true", help="print the configs and exit")
    p.add_argument("--algos", type=lambda s: s.split(","), default=None, help="e.g. dqn,ppo (default: all)")
    p.add_argument("--configs", type=lambda s: s.split(","), default=None,
                   help="indices (c00,c03) and/or full names (dqn_c08,ppo_c04) (default: all)")
    p.add_argument("--name-suffix", default="", help="appended to run names, e.g. _p2 for Phase 2")
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0], help="e.g. 0 or 1,2,3,4")
    p.add_argument("--total-frames", type=int, default=None, help="override the frame budget (default: 200k)")
    p.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    p.add_argument("--root", default=None, help="directory receiving logs/ and models/ (default: . or the dry-run dir)")
    p.add_argument("--tag", default=None, help="name of the status/results files (default: phase1 or dry_run)")
    p.add_argument("--dry-run", action="store_true", help=f"tiny budgets ({DRY_RUN_FRAMES} frames) in a scratch root")
    p.add_argument("--overwrite", action="store_true", help="re-run jobs whose summary.json has a different config")
    p.add_argument("--timeout", type=float, default=None, help="seconds before a single run is killed")
    args = p.parse_args(argv)

    entries = select_entries(args.algos, args.configs)
    if args.list:
        print(format_table(entries))
        return 0

    if args.dry_run:
        configs = [dry_run_config(e, s) for s in args.seeds for e in entries]
        root, tag = args.root or "logs/_scratch/sweep_dry", args.tag or "dry_run"
    else:
        configs = [build_config(e, s, args.total_frames, name_suffix=args.name_suffix)
                   for s in args.seeds for e in entries]
        root, tag = args.root or ".", args.tag or "phase1"
    print(f"sweep '{tag}' -> {Path(root).resolve()}")
    statuses = run_jobs(configs, root, workers=args.workers, overwrite=args.overwrite, tag=tag,
                        timeout_s=args.timeout)
    counts = {k: sum(s["status"] == k for s in statuses) for k in ("done", "skipped", "stale", "failed")}
    print("finished: " + ", ".join(f"{v} {k}" for k, v in counts.items())
          + f" | results: {Path(root) / SWEEP_DIR / (tag + '_results.csv')}")
    return 1 if counts["failed"] or counts["stale"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
