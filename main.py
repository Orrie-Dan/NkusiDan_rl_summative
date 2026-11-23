"""WasteFlow entry point: the command-line interface plus everything that runs experiments around the agents.

Layout (the brief's mandated structure): ``environment/`` is the world (simulation, Gymnasium env, rendering),
``training/`` trains one agent (``dqn_training.py``, ``pg_training.py``, shared infrastructure in ``__init__.py``),
and this file holds the comparison baselines, the evaluation harness, the hyperparameter sweep and the CLI.

Usage:
    uv run main.py                          # best trained agent (DQN) in the three.js viewer, same as play.py
    uv run main.py play --algo all          # all four trained agents one after another (flags: play.py)
    uv run main.py random-demo              # random agent in the three.js viewer (live, in the browser)
    uv run main.py replay logs/demo/random_episode.json   # play back a recorded episode
    uv run main.py train dqn --total-frames 100000 --name my_run   # train (flags: training/dqn_training.py)
    uv run main.py train reinforce --total-frames 100000           # (flags: training/pg_training.py)
    uv run main.py train a2c --total-frames 100000                 # (flags: training/pg_training.py)
    uv run main.py train ppo --total-frames 100000                 # (flags: training/pg_training.py)
    uv run main.py plots                    # rebuild report figures + tables in assets/figures/ from logs/
    uv run main.py sweep --list             # the 40 sweep configs
    uv run main.py sweep                    # Phase 1 sweep: 40 runs in parallel, resumable
    uv run main.py sweep --dry-run          # same 40 configs with tiny budgets, into logs/_scratch/
    uv run main.py evaluate                 # KPI table (mean + 95% CI) for the baselines
    uv run main.py evaluate --scenarios all --policies urgency_first,ppo=models/pg/<run>/model.pt
    uv run main.py --help                   # list commands
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from environment.custom_env import (
    ACTION_DISPOSAL,
    ACTION_WAIT,
    EVAL_SEEDS,
    FACILITY,
    N_LOCATIONS,
    Policy,
    REWARD_COMPONENTS,
    RewardConfig,
    run_episode,
    SCENARIOS,
    VehicleStatus,
    WasteFlowEnv,
    WasteFlowSim,
)
from training import ALGOS, log_dir_for, RunConfig
from training.dqn_training import DQNConfig
from training.pg_training import A2CConfig, PPOConfig, ReinforceConfig


# ====================================================================================================================
# Heuristic baselines (card 07)  (formerly baselines/heuristics.py)
# ====================================================================================================================
# Heuristic dispatch baselines (card 07): random, nearest-job-first and urgency-first.
#
# All three implement the ``environment.custom_env.Policy`` interface and only pick actions allowed by the same
# action mask the RL agents see. The two rule-based policies share one dispatcher skeleton so they differ only
# in how they pick the next bin:
#
# 1. Dispose when the truck is nearly full.
# 2. Near shift end, dispose (which ends at the facility/depot) so the truck is back and empty: the terminal
#    penalties punish waste left on trucks and trucks away from the depot.
# 3. Otherwise pick a bin with the policy's rule, skipping bins that can't be serviced and brought back
#    to disposal before the shift ends.
# 4. If no bin is worth visiting, dispose a part load or wait.
#
# Travel times are estimated from the layout distance, free-flow speed and the zones' current congestion
# (no noise), i.e. only from information that is also in the observation.
#
# Run ``uv run main.py evaluate`` to evaluate them (with the rolling-horizon planner) on the 50 eval seeds.

# Slack added to end-of-shift time estimates. It must cover one wait (up to 15 min), because a truck that
# waits instead of heading home only re-checks the clock when the wait ends.
END_OF_SHIFT_MARGIN_MIN = 20.0


def expected_travel_min(sim: WasteFlowSim, a: int, b: int) -> float:
    """Noise-free travel time estimate from node a to node b under the current congestion."""
    cfg = sim.config
    zones = sim.layout.node_zone
    mult = np.where(sim.zone_congested[[zones[a], zones[b]]], cfg.congestion_multiplier, 1.0).mean()
    return float(sim.layout.dist_km[a, b] / cfg.speed_kmh * 60.0 * mult)


def service_estimate_min(sim: WasteFlowSim, b: int) -> float:
    cfg = sim.config
    kg = sim.fill[b] * sim.layout.bin_capacity_kg[b]
    return cfg.service_base_min + cfg.service_min_per_tonne * kg / 1000.0


class RandomPolicy:
    """Uniform over the valid actions."""

    name = "random"

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict, env: WasteFlowEnv) -> int:
        return int(self._rng.choice(np.flatnonzero(obs["action_mask"])))


class _Dispatcher:
    """Shared skeleton for the rule-based policies; subclasses implement ``choose_bin``."""

    name = "dispatcher"
    dispose_at_load = 0.85  # dispose once the truck is this full
    idle_dispose_at_load = 0.30  # with nothing worth collecting, dispose a part load above this

    def reset(self, seed: int | None = None) -> None:
        pass  # deterministic

    def act(self, obs: dict, env: WasteFlowEnv) -> int:
        sim = env.sim
        v = sim.deciding_vehicle
        mask = np.asarray(obs["action_mask"], dtype=bool)
        node = int(sim.vehicle_node[v])
        load = sim.vehicle_load_kg[v] / sim.config.vehicle_capacity_kg
        time_left = sim.config.shift_minutes - sim.t
        can_dispose = mask[ACTION_DISPOSAL]

        if can_dispose and load >= self.dispose_at_load:
            return ACTION_DISPOSAL
        home = expected_travel_min(sim, node, FACILITY) + sim.config.disposal_min + END_OF_SHIFT_MARGIN_MIN
        if can_dispose and time_left <= home:
            return ACTION_DISPOSAL

        feasible = [b for b in np.flatnonzero(mask[:N_LOCATIONS]) if self._fits_in_shift(sim, node, int(b), time_left)]
        if feasible:
            choice = self.choose_bin(sim, node, np.asarray(feasible))
            if choice is not None:
                return int(choice)

        if can_dispose and load >= self.idle_dispose_at_load:
            return ACTION_DISPOSAL
        return ACTION_WAIT if mask[ACTION_WAIT] else ACTION_DISPOSAL

    @staticmethod
    def _fits_in_shift(sim: WasteFlowSim, node: int, b: int, time_left: float) -> bool:
        trip = (
            expected_travel_min(sim, node, b)
            + service_estimate_min(sim, b)
            + expected_travel_min(sim, b, FACILITY)
            + sim.config.disposal_min
        )
        return trip + END_OF_SHIFT_MARGIN_MIN <= time_left

    def choose_bin(self, sim: WasteFlowSim, node: int, candidates: np.ndarray) -> int | None:
        raise NotImplementedError


class NearestJobFirstPolicy(_Dispatcher):
    """Nearest bin whose fill is at or above ``fill_threshold``; dispose when (nearly) full."""

    name = "nearest_job_first"

    def __init__(self, fill_threshold: float = 0.5) -> None:
        self.fill_threshold = fill_threshold

    def choose_bin(self, sim: WasteFlowSim, node: int, candidates: np.ndarray) -> int | None:
        jobs = candidates[sim.fill[candidates] >= self.fill_threshold]
        if len(jobs) == 0:
            return None
        return int(jobs[np.argmin(sim.layout.dist_km[node, jobs])])


class UrgencyFirstPolicy(_Dispatcher):
    """Open urgent requests by earliest deadline (if still reachable in time), then the fullest bin."""

    name = "urgency_first"

    def __init__(self, min_fill: float = 0.3) -> None:
        self.min_fill = min_fill

    def choose_bin(self, sim: WasteFlowSim, node: int, candidates: np.ndarray) -> int | None:
        urgent = candidates[sim.urgent_pending[candidates]]
        if len(urgent):
            deadline_left = sim.urgent_deadline[urgent] - sim.t
            eta = np.array([expected_travel_min(sim, node, int(b)) for b in urgent])
            reachable = urgent[eta <= deadline_left]
            if len(reachable):
                return int(reachable[np.argmin(sim.urgent_deadline[reachable])])
        jobs = candidates[sim.fill[candidates] >= self.min_fill]
        if len(jobs) == 0:
            return None
        return int(jobs[np.argmax(sim.fill[jobs])])


HEURISTICS = {p.name: p for p in (RandomPolicy, NearestJobFirstPolicy, UrgencyFirstPolicy)}


# ====================================================================================================================
# Rolling-horizon optimization baseline (card 08)  (formerly baselines/rolling_horizon.py)
# ====================================================================================================================
# Optimization baseline (Trello card 08): rolling-horizon routing with cheapest insertion and 2-opt.
#
# At every decision of a vehicle:
#
# 1. Predict each bin's fill ``horizon_min`` ahead from its expected fill rate (base rate x demand multiplier x
#    time-of-day curve; the per-tick noise has mean 1 and is left out).
# 2. Candidates: open urgent requests still reachable before their deadline, then bins predicted to reach
#    ``target_fill`` within the horizon. If there are none, any bin at ``fallback_fill`` or more (as
#    nearest-job-first would collect it).
# 3. Build a route from the vehicle's position that ends at the facility: insert candidates (urgent first by
#    deadline, then by predicted time-to-full) at their cheapest position, keeping the route within the truck's
#    remaining capacity, the shift end and every urgent deadline on it.
# 4. Improve the stop order with 2-opt (shorter expected travel, still feasible).
# 5. Send the vehicle to the first stop. The plan is discarded and rebuilt at the next decision.
#
# Dispose-when-full, end-of-shift return and the fallback to part-load disposal or waiting come from the shared
# ``_Dispatcher`` skeleton of the heuristic baselines above, so only step 2-5 differ from the other rule-based policies.
# Travel times are noise-free estimates under the current congestion (``expected_travel_min``).

class RollingHorizonPolicy(_Dispatcher):
    name = "rolling_horizon"

    def __init__(self, horizon_min: float = 75.0, target_fill: float = 0.8, fallback_fill: float = 0.5,
                 max_candidates: int = 8) -> None:
        self.horizon_min = horizon_min
        self.target_fill = target_fill
        self.fallback_fill = fallback_fill
        self.max_candidates = max_candidates

    # ------------------------------------------------------------------------------------------- prediction

    def _rate_per_min(self, sim: WasteFlowSim) -> np.ndarray:
        cfg = sim.config
        mid = sim.t + self.horizon_min / 2
        return sim.layout.base_fill_rate * cfg.demand_multiplier * sim.demand_curve(mid) / 60.0

    def _candidates(self, sim: WasteFlowSim, node: int, feasible: np.ndarray) -> tuple[list[int], set[int]]:
        """Candidate bins in insertion order, and the subset that are urgent."""
        rate = self._rate_per_min(sim)
        urgent = [int(b) for b in feasible if sim.urgent_pending[b]
                  and sim.t + expected_travel_min(sim, node, int(b)) <= sim.urgent_deadline[b]]
        urgent.sort(key=lambda b: sim.urgent_deadline[b])
        predicted = sim.fill + rate * self.horizon_min
        due = [int(b) for b in feasible if int(b) not in urgent and predicted[b] >= self.target_fill]
        if not urgent and not due:
            due = [int(b) for b in feasible if sim.fill[b] >= self.fallback_fill]
        time_to_full = (1.0 - sim.fill) / np.maximum(rate, 1e-9)
        due.sort(key=lambda b: time_to_full[b])
        return (urgent + due)[: self.max_candidates], set(urgent)

    # ------------------------------------------------------------------------------------------------- routes

    def _kg_at(self, sim: WasteFlowSim, b: int, minutes_ahead: float, rate: np.ndarray) -> float:
        fill = min(sim.fill[b] + rate[b] * minutes_ahead, 1.0)
        return fill * sim.layout.bin_capacity_kg[b]

    def _evaluate(self, sim: WasteFlowSim, node: int, stops: list[int], urgent: set[int],
                  rate: np.ndarray) -> float | None:
        """Expected travel minutes of node -> stops -> facility, or None if the route is infeasible."""
        cfg = sim.config
        capacity = cfg.vehicle_capacity_kg - sim.vehicle_load_kg[sim.deciding_vehicle]
        t, travel, kg, here = 0.0, 0.0, 0.0, node
        for b in stops:
            leg = expected_travel_min(sim, here, b)
            t += leg
            travel += leg
            if b in urgent and sim.t + t > sim.urgent_deadline[b]:
                return None
            take = self._kg_at(sim, b, t, rate)
            kg += take
            if kg > capacity:
                return None
            t += cfg.service_base_min + cfg.service_min_per_tonne * take / 1000.0
            here = b
        leg = expected_travel_min(sim, here, FACILITY)
        t += leg + cfg.disposal_min
        if sim.t + t + END_OF_SHIFT_MARGIN_MIN > cfg.shift_minutes:
            return None
        return travel + leg

    def _insert(self, sim, node, candidates, urgent, rate) -> list[int]:
        route: list[int] = []
        for b in candidates:
            best, best_cost = None, None
            for pos in range(len(route) + 1):
                trial = route[:pos] + [b] + route[pos:]
                cost = self._evaluate(sim, node, trial, urgent, rate)
                if cost is not None and (best_cost is None or cost < best_cost):
                    best, best_cost = trial, cost
            if best is not None:
                route = best
        return route

    def _two_opt(self, sim, node, route, urgent, rate) -> list[int]:
        best_cost = self._evaluate(sim, node, route, urgent, rate)
        improved = True
        while improved and len(route) >= 2:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    trial = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    cost = self._evaluate(sim, node, trial, urgent, rate)
                    if cost is not None and cost < best_cost - 1e-9:
                        route, best_cost, improved = trial, cost, True
        return route

    def plan(self, sim: WasteFlowSim, node: int, candidates: np.ndarray) -> list[int]:
        """The full planned route (bin ids, facility implied at the end) for the deciding vehicle."""
        ordered, urgent = self._candidates(sim, node, candidates)
        rate = self._rate_per_min(sim)
        route = self._insert(sim, node, ordered, urgent, rate)
        return self._two_opt(sim, node, route, urgent, rate)

    def choose_bin(self, sim: WasteFlowSim, node: int, candidates: np.ndarray) -> int | None:
        route = self.plan(sim, node, candidates)
        return route[0] if route else None


# ====================================================================================================================
# Evaluation harness (card 16)  (formerly evaluation/harness.py)
# ====================================================================================================================
# Evaluation harness (Trello card 16): any policy on the same 50 fixed seeds, KPIs with 95% bootstrap CIs.
#
# Every policy (baseline or RL checkpoint) plays the same episodes, ``environment.custom_env.EVAL_SEEDS`` (common random
# numbers), in the chosen scenario(s). Stress tests are the other scenarios in ``environment.custom_env.SCENARIOS``.
#
# KPIs (one row per episode; ``run_episode`` from ``environment.custom_env`` plus the extras computed here):
#
# - ``return`` and its components ``r_collected``, ``r_travel``, ``r_overflow``, ``r_urgent_missed``,
#   ``r_terminal_load`` and ``r_terminal_not_returned`` (they sum to the return)
# - ``overflow_bin_minutes``: bin-minutes spent at 100 % fill
# - urgent requests: ``urgent_served`` (on time, i.e. before the deadline), ``urgent_missed``, ``urgent_service_rate``
# - ``bins_full_at_end``: missed collections, i.e. locations still overflowing when the shift ends
# - ``travel_minutes``: driving time of all vehicles
# - ``disposal_trips`` and ``avg_disposal_load_t``: tonnes per COMPLETED disposal (a disposal still in progress at
#   the shift end is left out of both the tonnes and the count)
# - ``vehicle_utilization``: share of vehicle-minutes the fleet is busy (travelling, servicing, disposing, or
#   broken down mid-trip), as opposed to waiting or out of service after a permanent breakdown
# - shift overrun: the simulation ends the shift on time, so overrun shows up as ``vehicles_not_returned`` (not
#   back at the depot when the shift ends) and ``tonnes_left_on_trucks``
#
# The extras are computed without touching the frozen env: ``RecordingPolicy`` logs (vehicle, time, action) for
# each decision, and the final simulation state is read once the episode ends.
#
# Outputs (``evaluate`` / CLI) in ``--out``:
#
# - ``episodes.csv``: one row per (policy, scenario, seed)
# - ``kpi_summary.csv``: per (policy, scenario, kpi): mean, std, 95% percentile-bootstrap CI, n
# - ``paired_vs_<reference>.csv``: per-seed paired difference (policy - reference) with a bootstrap CI, for the
#   headline KPIs. Pairing on seeds is valid because every policy plays the same episodes.
#
# Usage::
#
#     uv run main.py evaluate                                   # all baselines, base scenario
#     uv run main.py evaluate --scenarios all                   # + stress scenarios
#     uv run main.py evaluate --policies urgency_first,ppo=models/pg/ppo_ppo_c00_s0/model.pt --out logs/eval/x

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 0
DEFAULT_REFERENCE = "urgency_first"
DEFAULT_OUT = Path("logs") / "eval" / "baselines"
# Every non-learning baseline: the card 07 heuristics plus the card 08 rolling-horizon planner.
BASELINES = {**HEURISTICS, RollingHorizonPolicy.name: RollingHorizonPolicy}

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
    if spec in BASELINES:
        return BASELINES[spec]()
    label, _, path = spec.rpartition("=")
    path_ = Path(path)
    if not path_.is_file():
        raise ValueError(f"{spec!r} is neither a baseline ({', '.join(BASELINES)}) nor an existing checkpoint")
    from training import load_policy  # torch import only when a checkpoint is evaluated

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


def evaluate_main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate policies on the 50 fixed eval seeds (card 16)")
    p.add_argument("--policies", type=lambda s: s.split(","), default=list(BASELINES),
                   help="comma-separated baseline names and/or [label=]path/to/model.pt, or 'phase2' for all 20 "
                        "Phase 2 checkpoints (default: all baselines)")
    p.add_argument("--scenarios", default="base", help=f"comma-separated, or 'all' ({', '.join(SCENARIOS)})")
    p.add_argument("--episodes", type=int, default=len(EVAL_SEEDS), help="first N eval seeds (default: all 50)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="policy for paired differences ('' to skip)")
    p.add_argument("--selection", action="store_true",
                   help="use the first N model-selection seeds (30000+) instead of the test seeds")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    scenarios = list(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    if "phase2" in args.policies:
        from training import phase2_runs

        i = args.policies.index("phase2")
        args.policies[i:i + 1] = [f"{label}={path}" for label, path in phase2_runs().items()]
    if args.selection:
        from training import SELECTION_SEEDS

        seeds = list(SELECTION_SEEDS[: args.episodes])
    else:
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


# ====================================================================================================================
# Hyperparameter sweep (card 15)  (formerly training/sweep.py)
# ====================================================================================================================
# Hyperparameter sweep: 40 configs (10 per algorithm) and a parallel, resumable runner (Trello card 15).
#
# Configs (``SWEEP``): for each algorithm, config ``c00`` is the base config (the trainer dataclass defaults,
# tuned in cards 10-13) and ``c01``-``c09`` each change ONE factor from it (one-factor-at-a-time). Every entry
# carries the factor it varies and a short rationale for the report. All runs use the same frame budget
# (``RunConfig.total_frames``, 200k since env v1.0).
#
# Runner (``run_jobs``): every run is its own subprocess (``python -m training.<trainer> --config <json>``), so a
# crash, even a hard one, takes down only that run. A thread pool keeps ``workers`` subprocesses going.
# Resumable: a run whose ``summary.json`` exists with the same config is skipped. A summary with a DIFFERENT
# config is reported as ``stale`` and left alone unless ``overwrite=True``. Each run's stdout/stderr goes to
# ``<root>/logs/sweep/runs/<run_id>.log``. When the sweep ends, ``<root>/logs/sweep/<tag>_results.csv`` collects
# one row per run from the summaries, and ``<tag>_status.json`` records how every job ended.
#
# Usage::
#
#     uv run main.py sweep --list                    # print the 40 configs
#     uv run main.py sweep                           # Phase 1: 40 runs, seed 0, into logs/ and models/
#     uv run main.py sweep --dry-run                 # tiny budgets into logs/_scratch/sweep_dry/
#     uv run main.py sweep --configs dqn_c08,ppo_c04 --seeds 0,1,2,3,4 --total-frames 600000 --name-suffix _p2 --tag phase2  # Phase 2: picked configs x 5 seeds, own run names

PROJECT_ROOT = Path(__file__).resolve().parent
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


def sweep_main(argv=None) -> int:
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


# ====================================================================================================================
# Report figures and tables (card 20)
# ====================================================================================================================
# ``uv run main.py plots`` rebuilds every report figure and table in ``assets/figures/`` from result files only (no
# training, no environment rollouts):
#
# - ``logs/<algo>/<run_id>/train_log.csv`` and ``summary.json``: learning curves, DQN loss, PG entropy, convergence
# - ``logs/sweep/phase1_results.csv`` + ``logs/sweep/phase1_selection/``: Phase 1 sweep (40 configs, 200k frames) and
#   the re-scoring on selection seeds 30000-30049 that picked the Phase 2 configs
# - ``logs/sweep/phase2_results.csv``: which runs form Phase 2 (picked config x 5 seeds, 600k frames)
# - ``logs/eval/final/<scenario>/episodes.csv``: held-out test seeds, all scenarios (card 19)
#
# RL rows in the evaluation figures/tables pool the 5 Phase 2 seeds x 50 episodes; CIs are 95 % percentile bootstrap
# over episodes (``bootstrap_ci``). Learning curves are aligned on a 10k-frame grid (evaluation runs about every 10k
# frames); dense training columns are averaged per 10k-frame bin. Captions are written to ``captions.md``.

FIG_DIR = Path("assets") / "figures"
FINAL_EVAL_DIR = Path("logs") / "eval" / "final"
PLOT_ALGOS = ("dqn", "ppo", "a2c", "reinforce")  # report order (Phase 2 ranking); colours follow the algorithm
ALGO_LABEL = {"dqn": "DQN", "ppo": "PPO", "a2c": "A2C", "reinforce": "REINFORCE"}
ALGO_COLOR = {"dqn": "#2a78d6", "ppo": "#eb6834", "a2c": "#1baf7a", "reinforce": "#eda100"}  # validated, slots 1-4
BASELINE_ORDER = ("rolling_horizon", "urgency_first", "nearest_job_first", "random")
BASELINE_COLOR = "#b5b3ab"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CURVE_STEP = 10_000
SCENARIO_LABEL = {"base": "Base (training scenario)", "high_demand": "High demand (1.5x)",
                  "high_breakdown": "High breakdown rate", "heavy_traffic": "Heavy traffic"}
CONVERGENCE_THRESHOLDS = (40.0, 50.0)
# Bookkeeping fields left out of the base-hyperparameter tables
_NON_HYPER = {"algo", "name", "seed", "scenario", "total_frames", "eval_every_frames", "eval_episodes", "reward"}


def _plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "font.size": 9, "lines.linewidth": 2.0, "figure.dpi": 100, "savefig.dpi": 200,
    })


def _kframes(ax) -> None:
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.set_xlabel("environment frames")


def _binned(log: pd.DataFrame, col: str, step: int = CURVE_STEP) -> pd.Series:
    """``col`` averaged per ``step``-frame bin, indexed by the bin's start frame (NaN rows dropped)."""
    d = log[["frames", col]].dropna()
    return d.groupby((d["frames"] // step) * step)[col].mean()


def _seed_frame(runs: Sequence[pd.DataFrame], col: str) -> pd.DataFrame:
    """One column per run, rows = frame bins present in every run."""
    return pd.concat([_binned(r, col) for r in runs], axis=1, join="inner")


def _band(ax, wide: pd.DataFrame, color: str, label: str) -> tuple[float, float, str]:
    """Mean across runs as a line, min-max across runs as a band. Returns the line end for ``_end_labels``."""
    x, mean = wide.index.to_numpy(), wide.mean(axis=1).to_numpy()
    ax.fill_between(x, wide.min(axis=1), wide.max(axis=1), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, label=label)
    return float(x[-1]), float(mean[-1]), label


def _end_labels(ax, ends: Sequence[tuple[float, float, str]], min_gap: float = 0.05) -> None:
    """Direct labels at the line ends, pushed apart vertically (``min_gap`` = fraction of the y range)."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (x, y, text), ly in zip(ends, ys):
        ax.annotate(text, (x, y), xytext=(5, ly), textcoords=("offset points", "data"), va="center", fontsize=8,
                    color=INK_2, annotation_clip=False)


def _save(fig, out: Path, name: str, captions: dict[str, str], caption: str) -> None:
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    captions[name] = caption
    import matplotlib.pyplot as plt

    plt.close(fig)


def load_plot_data(root: Path = Path(".")) -> dict[str, Any]:
    """Everything the figures need, read from result files only."""
    p1 = pd.read_csv(root / SWEEP_DIR / "phase1_results.csv")
    p2 = pd.read_csv(root / SWEEP_DIR / "phase2_results.csv")
    sel = pd.read_csv(root / SWEEP_DIR / "phase1_selection" / "episodes.csv")
    phase2_names = {a: g["name"].iloc[0] for a, g in p2.groupby("algo")}
    missing = set(PLOT_ALGOS) - set(phase2_names)
    if missing:
        raise FileNotFoundError(f"phase2_results.csv has no runs for {sorted(missing)}")

    def log(algo: str, run_id: str) -> pd.DataFrame:
        return pd.read_csv(log_dir_for(algo, run_id, root) / "train_log.csv")

    p2_logs = {a: [log(a, f"{phase2_names[a]}_s{s}") for s in sorted(p2.loc[p2.algo == a, "seed"])]
               for a in PLOT_ALGOS}
    p1_logs = {a: {r["name"]: log(a, f"{r['name']}_s{r['seed']}") for _, r in p1[p1.algo == a].iterrows()}
               for a in PLOT_ALGOS}
    scen = [s for s in SCENARIOS if (root / FINAL_EVAL_DIR / s / "episodes.csv").exists()]
    if "base" not in scen:
        raise FileNotFoundError(f"no final evaluation in {root / FINAL_EVAL_DIR} (run card 19 first)")
    ep = pd.concat([pd.read_csv(root / FINAL_EVAL_DIR / s / "episodes.csv") for s in scen], ignore_index=True)
    ep["group"] = ep["policy"].str.replace(r"_s\d+$", "", regex=True)
    base_cfg = {a: json.loads((log_dir_for(a, f"{a}_c00_s0", root) / "config.json").read_text(encoding="utf-8"))
                for a in PLOT_ALGOS}
    p2_cfg = {a: json.loads((log_dir_for(a, f"{phase2_names[a]}_s0", root) / "config.json").read_text(encoding="utf-8"))
              for a in PLOT_ALGOS}
    return {"p1": p1, "p2": p2, "sel": sel, "phase2_names": phase2_names, "p2_logs": p2_logs, "p1_logs": p1_logs,
            "episodes": ep, "scenarios": scen, "base_cfg": base_cfg, "p2_cfg": p2_cfg}


# ------------------------------------------------------------------------------------------------ tables


def _ci_cell(m: float, lo: float, hi: float, fmt: str = ".1f") -> str:
    return f"{m:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def _policy_order(groups: Iterable[str]) -> list[str]:
    g = set(groups)
    return [p for p in (*BASELINE_ORDER, *PLOT_ALGOS) if p in g]


def _policy_label(p: str) -> str:
    return f"{ALGO_LABEL[p]} (5 seeds)" if p in ALGO_LABEL else p


def eval_table(ep: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Per policy group: return with CI, the seed spread for RL, and the headline KPIs (episode means)."""
    d = ep[ep["scenario"] == scenario].copy()
    d["terminal_penalty"] = d["r_terminal_load"] + d["r_terminal_not_returned"]
    rows = []
    for g in _policy_order(d["group"]):
        x = d[d["group"] == g]
        m, lo, hi = bootstrap_ci(x["return"])
        seed_means = x.groupby("policy")["return"].mean()
        rows.append({
            "policy": _policy_label(g), "return": m, "ci_low": lo, "ci_high": hi,
            "seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else float("nan"),
            "tonnes_collected": x["tonnes_collected"].mean(),
            "overflow_bin_minutes": x["overflow_bin_minutes"].mean(),
            "urgent_service_rate": x["urgent_service_rate"].mean(),
            "vehicles_not_returned": x["vehicles_not_returned"].mean(),
            "tonnes_left_on_trucks": x["tonnes_left_on_trucks"].mean(),
            "terminal_penalty": x["terminal_penalty"].mean(), "episodes": len(x),
        })
    return pd.DataFrame(rows)


def hyperparameter_table(data: dict[str, Any], algo: str) -> pd.DataFrame:
    """The algorithm's 10 Phase 1 configs with their outcomes (200k frames, seed 0)."""
    p1 = data["p1"][data["p1"].algo == algo].set_index("name")
    sel = data["sel"]
    picked = data["phase2_names"][algo].removesuffix("_p2")
    rows = []
    for e in SWEEP[algo]:
        r = p1.loc[e.name]
        m, lo, hi = bootstrap_ci(sel.loc[sel.policy == e.name, "return"])
        rows.append({
            "config": e.name, "factor": e.factor,
            "change": ", ".join(f"{k}={v}" for k, v in e.overrides.items()) or "(base)",
            "best_val_return": r["best_eval_return"], "final_val_return": r["final_eval_return"],
            "selection_return": m, "selection_ci": f"[{lo:.1f}, {hi:.1f}]",
            "final_overflow_bin_minutes": r["final_overflow_bin_minutes"],
            "final_urgent_service_rate": r["final_urgent_service_rate"],
            "picked": "yes" if e.name == picked else "", "rationale": e.rationale,
        })
    return pd.DataFrame(rows)


def generalization_table(data: dict[str, Any]) -> pd.DataFrame:
    """Per algorithm: best validation return (model selection) vs held-out test return, and return per scenario."""
    p2, ep = data["p2"], data["episodes"]
    rows = []
    for a in PLOT_ALGOS:
        val = p2.loc[p2.algo == a, "best_eval_return"]
        test = ep[(ep.group == a) & (ep.scenario == "base")].groupby("policy")["return"].mean()
        row = {"algorithm": ALGO_LABEL[a], "val_best_return": val.mean(), "test_return": test.mean(),
               "drop": val.mean() - test.mean()}
        for s in data["scenarios"]:
            row[f"return_{s}"] = ep[(ep.group == a) & (ep.scenario == s)]["return"].mean()
        rows.append(row)
    for b in BASELINE_ORDER:
        row = {"algorithm": b, "val_best_return": float("nan"), "test_return": float("nan"), "drop": float("nan")}
        for s in data["scenarios"]:
            row[f"return_{s}"] = ep[(ep.group == b) & (ep.scenario == s)]["return"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _write_table(df: pd.DataFrame, out: Path, name: str, title: str, floatfmt: str = ".2f") -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = [f"{v:{floatfmt}}" if isinstance(v, float) and not np.isnan(v) else ("" if isinstance(v, float)
                 else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------- figures


def fig_learning_curves(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, col in zip(axes, ("eval_return_mean", "train_return_mean")):
        ends = [_band(ax, _seed_frame(data["p2_logs"][a], col), ALGO_COLOR[a], ALGO_LABEL[a]) for a in PLOT_ALGOS]
        _end_labels(ax, ends)
    axes[0].set_title("(a) Evaluation return on 10 validation seeds")
    axes[1].set_title("(b) Training episode return (10k-frame bins)")
    axes[0].set_ylabel("episode return")
    for ax in axes:
        _kframes(ax)
    axes[0].legend(loc="lower right")
    _save(fig, out, "fig1_learning_curves", captions,
          "Learning curves of the four algorithms (Phase 2: picked config, 5 seeds, 600k frames). Line: mean across "
          "seeds; band: min-max across seeds. (a) Evaluation return on the 10 validation seeds in each agent's eval "
          "mode (DQN greedy, policy-gradient agents sampled). (b) Return of training episodes, averaged per "
          "10k-frame bin (includes DQN's epsilon exploration).")


def fig_dqn_loss(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    runs, c = data["p2_logs"]["dqn"], ALGO_COLOR["dqn"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    _band(axes[0], _seed_frame(runs, "loss"), c, "DQN")
    _band(axes[1], _seed_frame(runs, "q_value_mean"), c, "DQN")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) TD loss (log scale)")
    axes[0].set_ylabel("loss per gradient step")
    axes[1].set_title("(b) Mean predicted Q-value of sampled batches")
    axes[1].set_ylabel("Q-value")
    starts = data["p2_cfg"]["dqn"]["learning_starts"]
    for ax in axes:
        _kframes(ax)
        ax.axvline(starts, color=MUTED, linewidth=1, linestyle=":")
    axes[0].annotate("learning starts", (starts, 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK_2)
    _save(fig, out, "fig2_dqn_objective", captions,
          f"DQN objective ({data['phase2_names']['dqn'].removesuffix('_p2')}, 5 seeds). Line: mean across seeds; "
          "band: min-max. (a) TD loss per gradient step, averaged per 10k-frame bin. (b) Mean Q-value predicted for "
          "the sampled replay batches. The dotted line marks the first gradient update (learning_starts).")


def fig_pg_entropy(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    ends = [_band(ax, _seed_frame(data["p2_logs"][a], "entropy"), ALGO_COLOR[a], ALGO_LABEL[a])
            for a in ("ppo", "a2c", "reinforce")]
    ax.axhline(np.log(17), color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("ln 17: uniform over all 17 actions (upper bound)", (0, np.log(17)), xytext=(2, -11),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylim(bottom=0)
    _end_labels(ax, ends)
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Policy entropy during training")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    _kframes(ax)
    _save(fig, out, "fig3_pg_entropy", captions,
          "Entropy of the masked action distribution for the policy-gradient agents (Phase 2, 5 seeds), averaged per "
          "10k-frame bin. Line: mean across seeds; band: min-max. The dotted line is the entropy of a uniform policy "
          "over all 17 actions; with masking the real maximum is lower.")


def convergence_table(data) -> pd.DataFrame:
    """Per Phase 2 run: first evaluation frame at which validation return reaches each threshold (NaN = never)."""
    rows = []
    for a in PLOT_ALGOS:
        for s, log in enumerate(data["p2_logs"][a]):
            e = log.dropna(subset=["eval_return_mean"])
            row = {"algorithm": ALGO_LABEL[a], "seed": s}
            for t in CONVERGENCE_THRESHOLDS:
                hit = e.loc[e["eval_return_mean"] >= t, "frames"]
                row[f"frames_to_{t:g}"] = float(hit.iloc[0]) if len(hit) else float("nan")
            last = e[e["frames"] >= e["frames"].max() - 100_000]["eval_return_mean"]
            row["last100k_mean"], row["last100k_std"] = last.mean(), last.std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def fig_convergence(data, out, captions, conv: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    max_f = max(int(log["frames"].max()) for logs in data["p2_logs"].values() for log in logs)
    for ax, t in zip(axes[:2], CONVERGENCE_THRESHOLDS):
        col = f"frames_to_{t:g}"
        ticks = []
        for i, a in enumerate(PLOT_ALGOS):
            v = conv.loc[conv.algorithm == ALGO_LABEL[a], col].to_numpy()
            y = np.full(v.shape, i) + np.linspace(-0.15, 0.15, len(v))
            hit = ~np.isnan(v)
            ax.scatter(v[hit], y[hit], s=40, color=ALGO_COLOR[a], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if (~hit).any():
                ax.scatter(np.full((~hit).sum(), max_f * 1.04), y[~hit], s=40, facecolor="none",
                           edgecolor=ALGO_COLOR[a], linewidth=1.5, zorder=3)
            ticks.append(f"{ALGO_LABEL[a]}\n{hit.sum()}/{len(v)} seeds")
        ax.set_yticks(range(len(PLOT_ALGOS)), ticks)
        ax.invert_yaxis()
        ax.set_xlim(0, max_f * 1.07)
        ax.set_title(f"({'ab'[CONVERGENCE_THRESHOLDS.index(t)]}) Frames to validation return >= {t:g}")
        _kframes(ax)
        ax.grid(axis="y", visible=False)
    for a in PLOT_ALGOS:
        wide = _seed_frame(data["p2_logs"][a], "eval_return_mean")
        axes[2].plot(wide.index, wide.std(axis=1, ddof=1).rolling(5, min_periods=1).mean(), color=ALGO_COLOR[a],
                     label=ALGO_LABEL[a])
    axes[2].set_title("(c) Spread across seeds")
    axes[2].set_ylabel("std of validation return across 5 seeds")
    axes[2].legend(loc="upper right")
    _kframes(axes[2])
    _save(fig, out, "fig4_convergence", captions,
          "Convergence and stability (Phase 2, 5 seeds per algorithm). (a, b) First evaluation at which a run's "
          f"validation return reaches {CONVERGENCE_THRESHOLDS[0]:g} and {CONVERGENCE_THRESHOLDS[1]:g}; one dot per "
          "seed, hollow dots at the right edge never reached it. (c) Standard deviation of the validation return "
          "across seeds over training (5-evaluation rolling mean): lower means more stable across seeds.")


def _hbar_ci(ax, tbl: pd.DataFrame, value: str, lo: str | None = None, hi: str | None = None) -> None:
    y = np.arange(len(tbl))
    colors = [ALGO_COLOR.get(next((a for a in PLOT_ALGOS if p.startswith(ALGO_LABEL[a] + " ")), ""), BASELINE_COLOR)
              for p in tbl["policy"]]
    ax.barh(y, tbl[value], height=0.7, color=colors, edgecolor=SURFACE, linewidth=2)
    if lo and hi:
        ax.errorbar(tbl[value], y, xerr=[tbl[value] - tbl[lo], tbl[hi] - tbl[value]], fmt="none", ecolor=INK_2,
                    elinewidth=1, capsize=2.5)
    ax.set_yticks(y, tbl["policy"])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="y", visible=False)


def fig_generalization(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    scen = data["scenarios"]
    ncol = 2
    nrow = int(np.ceil(len(scen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow), sharex=True, squeeze=False, layout="constrained")
    ep = data["episodes"]
    for ax, s in zip(axes.flat, scen):
        tbl = eval_table(ep, s)
        _hbar_ci(ax, tbl, "return", "ci_low", "ci_high")
        d = ep[ep.scenario == s]
        for i, p in enumerate(tbl["policy"]):
            a = next((a for a in PLOT_ALGOS if p.startswith(ALGO_LABEL[a] + " ")), None)
            if a:
                seeds = d[d.group == a].groupby("policy")["return"].mean()
                ax.scatter(seeds, np.full(len(seeds), i), s=12, color=INK, zorder=4, linewidth=0)
        ax.set_title(SCENARIO_LABEL.get(s, s))
    for ax in axes.flat[len(scen):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("mean episode return on 50 held-out seeds")
    _save(fig, out, "fig5_generalization", captions,
          "Held-out evaluation: mean return on the 50 test seeds (never used for training or model selection) in "
          "the training scenario and three stress scenarios the agents never saw. Grey bars: baselines; coloured "
          "bars: RL agents pooled over 5 seeds x 50 episodes. Whiskers: 95% bootstrap CI; black dots: per-seed means.")


def fig_kpi_breakdown(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    d = data["episodes"]
    d = d[d.scenario == "base"].assign(terminal_penalty=lambda x: x.r_terminal_load + x.r_terminal_not_returned)
    panels = [("tonnes_collected", "(a) Tonnes collected", "tonnes per shift"),
              ("overflow_bin_minutes", "(b) Overflow", "bin-minutes at 100% fill"),
              ("terminal_penalty", "(c) End-of-shift penalty", "return lost (load on board + not at depot)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, layout="constrained")
    for ax, (k, title, xl) in zip(axes, panels):
        rows = []
        for g in _policy_order(d.group):
            m, lo, hi = bootstrap_ci(d.loc[d.group == g, k])
            rows.append({"policy": _policy_label(g), "v": m, "lo": lo, "hi": hi})
        _hbar_ci(ax, pd.DataFrame(rows), "v", "lo", "hi")
        ax.set_title(title)
        ax.set_xlabel(xl)
    _save(fig, out, "fig6_kpi_breakdown", captions,
          "Why the RL agents trail the heuristics (base scenario, 50 held-out seeds, 95% bootstrap CIs). They "
          "collect as much or more waste (a) but let bins overflow longer (b) and end the shift with trucks away "
          "from the depot or still loaded (c).")


def fig_phase1_configs(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    for ax, a in zip(axes.flat, PLOT_ALGOS):
        picked = data["phase2_names"][a].removesuffix("_p2")
        for name, log in data["p1_logs"][a].items():
            s = _binned(log, "eval_return_mean")
            if name == picked:
                continue
            base = name.endswith("_c00")
            ax.plot(s.index, s, color=INK_2 if base else AXIS, linewidth=1.5 if base else 1.0,
                    linestyle="--" if base else "-", zorder=2)
        s = _binned(data["p1_logs"][a][picked], "eval_return_mean")
        ax.plot(s.index, s, color=ALGO_COLOR[a], zorder=3)
        ax.set_title(f"{ALGO_LABEL[a]}: 10 configs")
        ax.legend(handles=[Line2D([], [], color=ALGO_COLOR[a], label=f"{picked} (picked)"),
                           Line2D([], [], color=INK_2, linestyle="--", linewidth=1.5, label="c00 (base)"),
                           Line2D([], [], color=AXIS, linewidth=1, label="other one-factor variants")],
                  loc="lower right")
        _kframes(ax)
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation return")
    _save(fig, out, "fig7_phase1_configs", captions,
          "Phase 1 sweep: validation-return curves of all 10 configs per algorithm (seed 0, 200k frames). Dashed: "
          "base config c00; grey: the nine one-factor variants; coloured: the config picked for Phase 2 (by re-"
          "scoring on 50 separate selection seeds, see the hyperparameter tables).")


def fig_sensitivity(data, out, captions) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained")
    for ax, a in zip(axes.flat, PLOT_ALGOS):
        t = hyperparameter_table(data, a)
        lo = t["selection_ci"].str.extract(r"\[(.*), (.*)\]").astype(float)
        y = np.arange(len(t))
        colors = [ALGO_COLOR[a] if p else BASELINE_COLOR for p in t["picked"]]
        ax.barh(y, t["selection_return"], height=0.7, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(t["selection_return"], y, xerr=[t["selection_return"] - lo[0], lo[1] - t["selection_return"]],
                    fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5)
        ax.set_yticks(y, [f"{c[-3:]}  {ch}" for c, ch in zip(t["config"], t["change"])], fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(t["selection_return"].iloc[0], color=INK_2, linewidth=1, linestyle="--")
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{ALGO_LABEL[a]}")
    for ax in axes[-1]:
        ax.set_xlabel("mean return on 50 selection seeds (200k frames)")
    _save(fig, out, "fig8_sensitivity", captions,
          "Hyperparameter sensitivity: each Phase 1 config re-scored on 50 selection seeds (30000-30049, disjoint "
          "from training, validation and test seeds). Coloured: picked for Phase 2; dashed line: base config c00; "
          "whiskers: 95% bootstrap CI.")


def make_plots(root: Path = Path("."), out: Path | None = None) -> list[Path]:
    """Rebuild all figures and tables; returns the files written."""
    _plot_style()
    out = Path(out) if out is not None else root / FIG_DIR
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    data = load_plot_data(root)
    captions: dict[str, str] = {}

    fig_learning_curves(data, out, captions)
    fig_dqn_loss(data, out, captions)
    fig_pg_entropy(data, out, captions)
    conv = convergence_table(data)
    fig_convergence(data, out, captions, conv)
    fig_generalization(data, out, captions)
    fig_kpi_breakdown(data, out, captions)
    fig_phase1_configs(data, out, captions)
    fig_sensitivity(data, out, captions)

    for a in PLOT_ALGOS:
        _write_table(hyperparameter_table(data, a), tables, f"hyperparameters_{a}",
                     f"{ALGO_LABEL[a]}: Phase 1 configs (200k frames, seed 0)")
        cfg = {k: v for k, v in data["base_cfg"][a].items() if k not in _NON_HYPER}
        _write_table(pd.DataFrame({"parameter": list(cfg), "value": [json.dumps(v) if isinstance(v, (list, dict))
                                                                      else str(v) for v in cfg.values()]}),
                     tables, f"base_config_{a}", f"{ALGO_LABEL[a]}: base config c00")
    for s in data["scenarios"]:
        _write_table(eval_table(data["episodes"], s), tables, f"final_eval_{s}",
                     f"Held-out evaluation, {s} (50 test seeds; RL pooled over 5 seeds)")
    _write_table(generalization_table(data), tables, "generalization",
                 "Validation (model selection) vs held-out test return, and return per scenario")
    _write_table(conv, tables, "convergence", "Frames to reach validation-return thresholds (Phase 2 runs)",
                 floatfmt=".1f")

    lines = ["# Figure captions", "", "Generated by `uv run main.py plots` from the files in `logs/`.", ""]
    lines += [f"**{name}.png**: {text}\n" for name, text in captions.items()]
    (out / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    return sorted(p for p in out.rglob("*") if p.is_file())


def plots_main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Regenerate report figures and tables from logs/ (card 20)")
    p.add_argument("--root", type=Path, default=Path("."), help="project root holding logs/")
    p.add_argument("--out", type=Path, default=None, help=f"output directory (default: <root>/{FIG_DIR})")
    args = p.parse_args(argv)
    files = make_plots(args.root, args.out)
    print(f"{len(files)} files written to {args.out or args.root / FIG_DIR}")
    return 0


# ====================================================================================================================
# Command-line interface  (formerly main.py)
# ====================================================================================================================

COMMANDS = {
    "play": "Play the best trained agent(s) in the three.js viewer (card 21; flags: play.py --help)",
    "train": "Train one algorithm (card 10-13)",
    "sweep": "Run the 40-config hyperparameter sweep (card 15)",
    "evaluate": "Evaluate agents and baselines on fixed seeds (card 16)",
    "plots": "Regenerate report figures from logs/ (card 20)",
    "random-demo": "Run a random agent in the three.js viewer (card 05)",
    "replay": "Play back a recorded episode JSON in the three.js viewer (card 05)",
}
PASSTHROUGH = ("play", "sweep", "evaluate", "plots")  # commands whose flags are parsed by their own module


def print_status() -> None:
    import gymnasium
    import tensordict
    import torch
    import torchrl

    print("WasteFlow - Intelligent Waste Operations & Fleet Optimization")
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | torchrl {torchrl.__version__} "
          f"| tensordict {tensordict.__version__} | gymnasium {gymnasium.__version__}")
    print("Playing the best trained agent (`uv run main.py --help` lists the other commands).")


def random_demo(args: argparse.Namespace) -> int:
    from environment.custom_env import WasteFlowEnv

    env = WasteFlowEnv(
        args.scenario,
        verbose=True,
        render_mode="human",
        render_options={"port": args.port, "speed": args.speed, "open_browser": not args.no_browser},
    )
    policy = RandomPolicy()
    policy.reset(args.seed)
    try:
        obs, _ = env.reset(seed=args.seed)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(policy.act(obs, env))
        if args.record:
            print(f"episode saved to {env.save_episode(args.record)}")
        if args.hold > 0:
            import time

            print(f"keeping the viewer up for {args.hold:.0f} s (Ctrl+C to quit)")
            time.sleep(args.hold)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    return 0


def replay(args: argparse.Namespace) -> int:
    import time
    import webbrowser

    from environment.rendering import ViewerServer

    server = ViewerServer(port=args.port, replay_path=args.path)
    url = server.replay_url
    print(f"WasteFlow replay of {args.path}: {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="WasteFlow RL summative")
    sub = parser.add_subparsers(dest="command")
    # passthrough commands answer --help themselves, so their subparsers have no -h of their own
    parsers = {name: sub.add_parser(name, help=help_text, add_help=name not in PASSTHROUGH)
               for name, help_text in COMMANDS.items()}

    demo = parsers["random-demo"]
    demo.add_argument("--scenario", default="base")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--speed", type=float, default=10.0, help="simulated minutes per real second")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--record", help="save the episode JSON here (for replay mode)")
    demo.add_argument("--hold", type=float, default=30.0, help="seconds to keep the viewer up after the shift")
    demo.add_argument("--no-browser", action="store_true", help="don't open a browser (open the URL yourself)")

    rep = parsers["replay"]
    rep.add_argument("path", help="episode JSON written by random-demo --record")
    rep.add_argument("--port", type=int, default=8766)
    rep.add_argument("--no-browser", action="store_true")

    tr = parsers["train"]
    tr.add_argument("algo", choices=["dqn", "reinforce", "a2c", "ppo"], help="algorithm to train")
    tr.add_argument("flags", nargs=argparse.REMAINDER, help="trainer flags, e.g. --total-frames 100000 --seed 1")

    # passthrough commands get every flag (REMAINDER cannot capture flags that come first)
    args, extra = parser.parse_known_args()
    if args.command in (None, "play"):
        import play

        if args.command is None:
            if extra:
                parser.error(f"unrecognized arguments: {' '.join(extra)}")
            print_status()
        return play.main(extra)
    if args.command == "sweep":
        return sweep_main(extra)
    if args.command == "evaluate":
        return evaluate_main(extra)
    if args.command == "plots":
        return plots_main(extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "random-demo":
        return random_demo(args)
    if args.command == "replay":
        return replay(args)
    if args.command == "train":
        if args.algo == "dqn":
            from training import dqn_training

            return dqn_training.main(args.flags)
        from training import pg_training

        return pg_training.main([args.algo, *args.flags])

    print(f"'{args.command}' is not implemented yet: {COMMANDS[args.command]}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
