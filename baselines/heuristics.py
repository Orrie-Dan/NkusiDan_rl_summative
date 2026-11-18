"""Heuristic dispatch baselines (card 07): random, nearest-job-first and urgency-first.

All three implement the ``evaluation.rollout.Policy`` interface and only pick actions allowed by the same
action mask the RL agents see. The two rule-based policies share one dispatcher skeleton so they differ only
in how they pick the next bin:

1. Dispose when the truck is nearly full.
2. Near shift end, dispose (which ends at the facility/depot) so the truck is back and empty: the terminal
   penalties punish waste left on trucks and trucks away from the depot.
3. Otherwise pick a bin with the policy's rule, skipping bins that can't be serviced and brought back
   to disposal before the shift ends.
4. If no bin is worth visiting, dispose a part load or wait.

Travel times are estimated from the layout distance, free-flow speed and the zones' current congestion
(no noise), i.e. only from information that is also in the observation.

Run ``uv run python -m baselines.heuristics`` to evaluate all three on the 50 eval seeds and write
``logs/baselines/kpis_<scenario>.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from environment.custom_env import WasteFlowEnv
from environment.simulation import ACTION_DISPOSAL, ACTION_WAIT, FACILITY, N_LOCATIONS, SCENARIOS, WasteFlowSim

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


def main() -> None:
    import pandas as pd

    from evaluation.rollout import EVAL_SEEDS, run_policy

    parser = argparse.ArgumentParser(description="Evaluate the heuristic baselines on the fixed eval seeds")
    parser.add_argument("--scenario", default="base", choices=list(SCENARIOS))
    parser.add_argument("--out", type=Path, default=Path("logs/baselines"))
    args = parser.parse_args()

    rows = []
    for cls in HEURISTICS.values():
        rows += run_policy(cls(), args.scenario, EVAL_SEEDS)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"kpis_{args.scenario}.csv"
    df.to_csv(path, index=False)

    cols = ["return", "tonnes_collected", "travel_minutes", "overflow_bin_minutes", "urgent_service_rate",
            "tonnes_left_on_trucks", "vehicles_not_returned"]
    summary = df.groupby("policy", sort=False)[cols].agg(["mean", "std"]).round(2)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"{args.scenario}: {len(EVAL_SEEDS)} episodes per policy -> {path}")
        print(summary)


if __name__ == "__main__":
    main()
