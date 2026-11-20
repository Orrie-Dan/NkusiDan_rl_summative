"""Optimization baseline (Trello card 08): rolling-horizon routing with cheapest insertion and 2-opt.

At every decision of a vehicle:

1. Predict each bin's fill ``horizon_min`` ahead from its expected fill rate (base rate x demand multiplier x
   time-of-day curve; the per-tick noise has mean 1 and is left out).
2. Candidates: open urgent requests still reachable before their deadline, then bins predicted to reach
   ``target_fill`` within the horizon. If there are none, any bin at ``fallback_fill`` or more (as
   nearest-job-first would collect it).
3. Build a route from the vehicle's position that ends at the facility: insert candidates (urgent first by
   deadline, then by predicted time-to-full) at their cheapest position, keeping the route within the truck's
   remaining capacity, the shift end and every urgent deadline on it.
4. Improve the stop order with 2-opt (shorter expected travel, still feasible).
5. Send the vehicle to the first stop. The plan is discarded and rebuilt at the next decision.

Dispose-when-full, end-of-shift return and the fallback to part-load disposal or waiting come from the shared
``_Dispatcher`` skeleton in ``baselines.heuristics``, so only step 2-5 differ from the other rule-based policies.
Travel times are noise-free estimates under the current congestion (``expected_travel_min``).
"""

from __future__ import annotations

import numpy as np

from baselines.heuristics import END_OF_SHIFT_MARGIN_MIN, _Dispatcher, expected_travel_min
from environment.simulation import FACILITY, WasteFlowSim


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
