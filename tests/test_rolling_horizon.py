"""Card 08: rolling-horizon insertion + 2-opt baseline.

Done-when (< 50 ms per decision, comparable to nearest-job-first on the base scenario) was checked with the
50-seed harness run; these tests cover validity and the planner's route constraints on a few seeds.
"""

import numpy as np
import pytest

from main import RollingHorizonPolicy
from environment.custom_env import REWARD_COMPONENTS, WasteFlowEnv
from environment.custom_env import N_LOCATIONS
from main import BASELINES, evaluate_policy, make_policy
from environment.custom_env import EVAL_SEEDS


def test_registered_and_plays_valid_full_episodes():
    assert isinstance(make_policy("rolling_horizon"), RollingHorizonPolicy) and "rolling_horizon" in BASELINES
    rows = evaluate_policy(RollingHorizonPolicy(), "base", EVAL_SEEDS[:3])  # the env raises on invalid actions
    for r in rows:
        assert sum(r[f"r_{k}"] for k in REWARD_COMPONENTS) == pytest.approx(r["return"])
        assert r["tonnes_collected"] > 10 and r["vehicles_not_returned"] <= 1


def test_deterministic():
    a = evaluate_policy(RollingHorizonPolicy(), "base", EVAL_SEEDS[:2])
    b = evaluate_policy(RollingHorizonPolicy(), "base", EVAL_SEEDS[:2])
    assert a == b


def _first_decision(seed=EVAL_SEEDS[0]):
    env = WasteFlowEnv("base")
    obs, _ = env.reset(seed=seed)
    return env, env.sim, np.flatnonzero(obs["action_mask"][:N_LOCATIONS])


def test_planned_route_is_feasible_and_two_opt_never_worsens_it():
    pol = RollingHorizonPolicy()
    env, sim, feasible = _first_decision()
    sim.fill[:] = 0.9  # many candidates, so capacity limits the route
    node = int(sim.vehicle_node[sim.deciding_vehicle])
    ordered, urgent = pol._candidates(sim, node, feasible)
    rate = pol._rate_per_min(sim)
    inserted = pol._insert(sim, node, ordered, urgent, rate)
    improved = pol._two_opt(sim, node, list(inserted), urgent, rate)
    assert len(inserted) >= 2 and sorted(improved) == sorted(inserted)
    assert pol._evaluate(sim, node, improved, urgent, rate) <= pol._evaluate(sim, node, inserted, urgent, rate)
    # capacity binds: adding any left-out candidate makes the route infeasible
    left_out = [b for b in ordered if b not in improved]
    assert left_out and all(pol._evaluate(sim, node, improved + [b], urgent, rate) is None for b in left_out)
    assert pol.plan(sim, node, feasible)[0] == pol.choose_bin(sim, node, feasible)


def test_routes_violating_deadlines_or_the_shift_end_are_rejected():
    pol = RollingHorizonPolicy()
    env, sim, feasible = _first_decision()
    node = int(sim.vehicle_node[sim.deciding_vehicle])
    rate = pol._rate_per_min(sim)
    b = int(feasible[0])
    assert pol._evaluate(sim, node, [b], set(), rate) is not None
    sim.urgent_deadline[b] = sim.t + 0.1  # cannot get there in 6 seconds
    assert pol._evaluate(sim, node, [b], {b}, rate) is None
    sim.t = sim.config.shift_minutes - 5.0  # no time left to visit and get home
    assert pol._evaluate(sim, node, [b], set(), rate) is None
