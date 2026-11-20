"""Evaluation harness (Trello card 16): any policy on the same 50 fixed seeds, KPIs with 95% bootstrap CIs.

Every policy (baseline or RL checkpoint) plays the same episodes, ``evaluation.rollout.EVAL_SEEDS`` (common random
numbers), in the chosen scenario(s). Stress tests are the other scenarios in ``environment.simulation.SCENARIOS``.

KPIs (one row per episode; ``run_episode`` from ``evaluation.rollout`` plus the extras computed here):

- ``return`` and its components ``r_collected``, ``r_travel``, ``r_overflow``, ``r_urgent_missed``,
  ``r_terminal_load`` and ``r_terminal_not_returned`` (they sum to the return)
- ``overflow_bin_minutes``: bin-minutes spent at 100 % fill
- urgent requests: ``urgent_served`` (on time, i.e. before the deadline), ``urgent_missed``, ``urgent_service_rate``
- ``bins_full_at_end``: missed collections, i.e. locations still overflowing when the shift ends
- ``travel_minutes``: driving time of all vehicles
- ``disposal_trips`` and ``avg_disposal_load_t``: tonnes per COMPLETED disposal (a disposal still in progress at
  the shift end is left out of both the tonnes and the count)
- ``vehicle_utilization``: share of vehicle-minutes the fleet is busy (travelling, servicing, disposing, or
  broken down mid-trip), as opposed to waiting or out of service after a permanent breakdown
- shift overrun: the simulation ends the shift on time, so overrun shows up as ``vehicles_not_returned`` (not
  back at the depot when the shift ends) and ``tonnes_left_on_trucks``

The extras are computed without touching the frozen env: ``RecordingPolicy`` logs (vehicle, time, action) for
each decision, and the final simulation state is read once the episode ends.

Outputs (``evaluate`` / CLI) in ``--out``:

- ``episodes.csv``: one row per (policy, scenario, seed)
- ``kpi_summary.csv``: per (policy, scenario, kpi): mean, std, 95% percentile-bootstrap CI, n
- ``paired_vs_<reference>.csv``: per-seed paired difference (policy - reference) with a bootstrap CI, for the
  headline KPIs. Pairing on seeds is valid because every policy plays the same episodes.

Usage::

    uv run main.py evaluate                                   # all baselines, base scenario
    uv run main.py evaluate --scenarios all                   # + stress scenarios
    uv run main.py evaluate --policies urgency_first,ppo=models/pg/ppo_ppo_c00_s0/model.pt --out logs/eval/x
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.heuristics import HEURISTICS
from environment.custom_env import REWARD_COMPONENTS, RewardConfig, WasteFlowEnv
from environment.simulation import ACTION_WAIT, SCENARIOS, VehicleStatus
from evaluation.rollout import EVAL_SEEDS, Policy, run_episode

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 0
DEFAULT_REFERENCE = "urgency_first"
DEFAULT_OUT = Path("logs") / "eval" / "baselines"

# Summary KPIs, in table order. "Higher is better" matters only for reading; nothing here ranks policies.
KPIS: tuple[str, ...] = (
    "return", *(f"r_{k}" for k in REWARD_COMPONENTS),
    "tonnes_collected", "overflow_bin_minutes", "urgent_served", "urgent_missed", "urgent_service_rate",
    "bins_full_at_end", "travel_minutes", "disposal_trips", "avg_disposal_load_t", "vehicle_utilization",
    "vehicles_not_returned", "tonnes_left_on_trucks", "breakdowns", "decisions",
)
PAIRED_KPIS: tuple[str, ...] = (
    "return", "tonnes_collected", "overflow_bin_minutes", "urgent_service_rate", "bins_full_at_end",
    "travel_minutes", "vehicles_not_returned",
)


# ------------------------------------------------------------------------------------------------- policies


class RecordingPolicy:
    """Wraps a policy and records ``(vehicle, sim time, action)`` for every decision it makes."""

    def __init__(self, inner: Policy) -> None:
        self.inner = inner
        self.name = inner.name
        self.decisions: list[tuple[int, float, int]] = []

    def reset(self, seed: int | None = None) -> None:
        self.decisions = []
        self.inner.reset(seed)

    def act(self, obs: dict, env: WasteFlowEnv) -> int:
        action = int(self.inner.act(obs, env))
        self.decisions.append((int(env.sim.deciding_vehicle), float(env.sim.t), action))
        return action


def make_policy(spec: str) -> Policy:
    """Policy from a spec: a baseline name (``urgency_first``), a checkpoint path, or ``label=path``."""
    if spec in HEURISTICS:
        return HEURISTICS[spec]()
    label, _, path = spec.rpartition("=")
    path_ = Path(path)
    if not path_.is_file():
        raise ValueError(f"{spec!r} is neither a baseline ({', '.join(HEURISTICS)}) nor an existing checkpoint")
    from training.common import load_policy  # torch import only when a checkpoint is evaluated

    # model.pt lives in models/dqn/<run_id>/ or models/pg/<algo>_<run_id>/: the folder names the run
    return load_policy(path_, name=label or path_.parent.name)


# ---------------------------------------------------------------------------------------------- episode KPIs


def extra_kpis(env: WasteFlowEnv, decisions: Sequence[tuple[int, float, int]]) -> dict:
    """KPIs that ``run_episode`` does not report, from the decision log and the final simulation state."""
    sim = env.sim
    cfg = sim.config
    shift = cfg.shift_minutes

    busy = 0.0
    for v in range(cfg.n_vehicles):
        own = sorted((t, a) for veh, t, a in decisions if veh == v)
        for i, (t, a) in enumerate(own):
            end = own[i + 1][0] if i + 1 < len(own) else shift
            if i + 1 == len(own) and sim.vehicle_out_of_service[v]:
                end = min(end, float(sim.route_arrive_at[v]))  # permanent breakdown time
            if a != ACTION_WAIT:
                busy += max(end - t, 0.0)

    disposing_at_end = int((sim.vehicle_status == VehicleStatus.DISPOSING).sum())
    completed = sim.stats.disposal_trips - disposing_at_end
    return {
        "bins_full_at_end": int((sim.fill >= 1.0).sum()),
        "avg_disposal_load_t": sim.stats.kg_disposed / 1000.0 / completed if completed > 0 else float("nan"),
        "vehicle_utilization": busy / (cfg.n_vehicles * shift),
    }


def evaluate_policy(policy: Policy, scenario: str = "base", seeds: Iterable[int] = EVAL_SEEDS,
                    reward_config: RewardConfig | None = None) -> list[dict]:
    """KPI rows (``run_episode`` + ``extra_kpis``) for ``policy`` on every seed."""
    env = WasteFlowEnv(scenario, reward_config=reward_config)
    rec = RecordingPolicy(policy)
    rows = []
    for seed in seeds:
        row = run_episode(env, rec, seed)
        row.update(extra_kpis(env, rec.decisions))
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ statistics


def bootstrap_ci(values, n_boot: int = N_BOOTSTRAP, level: float = CI_LEVEL,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap CI of the mean. NaNs are dropped; all-NaN gives NaNs."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(x.mean()), float(lo), float(hi)


def summarize(episodes: pd.DataFrame, kpis: Sequence[str] = KPIS) -> pd.DataFrame:
    """Long table: policy, scenario, kpi, mean, std, ci_low, ci_high, n (policy order as in ``episodes``)."""
    rows = []
    for (policy, scenario), g in episodes.groupby(["policy", "scenario"], sort=False):
        for k in kpis:
            mean, lo, hi = bootstrap_ci(g[k])
            rows.append({"policy": policy, "scenario": scenario, "kpi": k, "mean": mean,
                         "std": float(g[k].std(ddof=1)) if g[k].notna().sum() > 1 else float("nan"),
                         "ci_low": lo, "ci_high": hi, "n": int(g[k].notna().sum())})
    return pd.DataFrame(rows)


def paired_differences(episodes: pd.DataFrame, reference: str,
                       kpis: Sequence[str] = PAIRED_KPIS) -> pd.DataFrame:
    """Per (policy, scenario, kpi): mean of (policy - reference) over shared seeds, with a bootstrap CI."""
    rows = []
    for scenario, sg in episodes.groupby("scenario", sort=False):
        ref = sg[sg["policy"] == reference].set_index("seed")
        if ref.empty:
            continue
        for policy, g in sg.groupby("policy", sort=False):
            if policy == reference:
                continue
            g = g.set_index("seed")
            shared = g.index.intersection(ref.index)
            for k in kpis:
                mean, lo, hi = bootstrap_ci(g.loc[shared, k] - ref.loc[shared, k])
                rows.append({"policy": policy, "reference": reference, "scenario": scenario, "kpi": k,
                             "mean_diff": mean, "ci_low": lo, "ci_high": hi, "n": len(shared),
                             "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- driver


@dataclass
class EvalResult:
    episodes: pd.DataFrame
    summary: pd.DataFrame
    paired: pd.DataFrame


def evaluate(policy_specs: Sequence[str], scenarios: Sequence[str] = ("base",), seeds: Iterable[int] = EVAL_SEEDS,
             reference: str | None = DEFAULT_REFERENCE, out: str | Path | None = None) -> EvalResult:
    """Evaluate every policy in every scenario on ``seeds``; write the CSVs to ``out`` when given."""
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown scenarios {sorted(unknown)}; expected {list(SCENARIOS)}")
    seeds = tuple(seeds)
    policies = [make_policy(spec) for spec in policy_specs]
    names = [pol.name for pol in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"policy names must be unique (use label=path for checkpoints), got {names}")
    rows = []
    for policy in policies:
        for scenario in scenarios:
            rows += evaluate_policy(policy, scenario, seeds)
    episodes = pd.DataFrame(rows)
    summary = summarize(episodes)
    paired = paired_differences(episodes, reference) if reference in names else pd.DataFrame()

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(out / "episodes.csv", index=False)
        summary.to_csv(out / "kpi_summary.csv", index=False)
        if not paired.empty:
            paired.to_csv(out / f"paired_vs_{reference}.csv", index=False)
    return EvalResult(episodes, summary, paired)


def format_summary(summary: pd.DataFrame, kpis: Sequence[str] = KPIS) -> str:
    """Wide text table: one column per policy/scenario, cells ``mean [low, high]``."""
    s = summary[summary["kpi"].isin(kpis)].copy()
    s["cell"] = [f"{m:.2f} [{lo:.2f}, {hi:.2f}]" for m, lo, hi in zip(s["mean"], s["ci_low"], s["ci_high"])]
    s["column"] = s["policy"] + " | " + s["scenario"]
    wide = s.pivot(index="kpi", columns="column", values="cell")
    wide = wide.reindex([k for k in kpis if k in wide.index])[list(dict.fromkeys(s["column"]))]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 30):
        return wide.to_string()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate policies on the 50 fixed eval seeds (card 16)")
    p.add_argument("--policies", type=lambda s: s.split(","), default=list(HEURISTICS),
                   help="comma-separated baseline names and/or [label=]path/to/model.pt (default: all baselines)")
    p.add_argument("--scenarios", default="base", help=f"comma-separated, or 'all' ({', '.join(SCENARIOS)})")
    p.add_argument("--episodes", type=int, default=len(EVAL_SEEDS), help="first N eval seeds (default: all 50)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="policy for paired differences ('' to skip)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    scenarios = list(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    seeds = EVAL_SEEDS[: args.episodes]
    try:
        res = evaluate(args.policies, scenarios, seeds, args.reference or None, args.out)
    except ValueError as exc:  # bad policy spec, duplicate name or unknown scenario
        p.error(str(exc))
    print(f"{len(args.policies)} policies x {len(scenarios)} scenarios x {len(seeds)} episodes -> {args.out}")
    print(format_summary(res.summary))
    if not res.paired.empty:
        p_ret = res.paired[res.paired["kpi"] == "return"]
        print(f"\npaired return difference vs {args.reference} (95% CI):")
        for r in p_ret.itertuples():
            print(f"  {r.policy:20s} {r.scenario:15s} {r.mean_diff:+8.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
                  f"{'  *' if r.significant else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
