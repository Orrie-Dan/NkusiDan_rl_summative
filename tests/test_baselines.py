"""Card 07: heuristic baselines.

Done-when: all three policies run 50 episodes and produce KPI rows.
"""

import numpy as np
import pandas as pd
import pytest

from main import HEURISTICS, NearestJobFirstPolicy, RandomPolicy, UrgencyFirstPolicy
from environment.custom_env import REWARD_COMPONENTS, WasteFlowEnv
from environment.custom_env import SCENARIOS
from environment.custom_env import EVAL_SEEDS, run_episode, run_policy

KPI_KEYS = {
    "policy", "scenario", "seed", "return", "tonnes_collected", "travel_minutes", "overflow_bin_minutes",
    "urgent_service_rate", "tonnes_left_on_trucks", "vehicles_not_returned",
}


class MaskCheckingPolicy:
    """Wraps a policy and asserts every chosen action is allowed by the observation's mask."""

    def __init__(self, inner):
        self.inner, self.name, self.calls = inner, inner.name, 0

    def reset(self, seed=None):
        self.inner.reset(seed)

    def act(self, obs, env):
        a = self.inner.act(obs, env)
        assert obs["action_mask"][a], f"{self.name} picked masked action {a}"
        self.calls += 1
        return a


@pytest.fixture(scope="module")
def base_kpis():
    rows = []
    for cls in HEURISTICS.values():
        rows += run_policy(MaskCheckingPolicy(cls()), "base", EVAL_SEEDS)
    return pd.DataFrame(rows)


def test_all_three_run_50_episodes_and_produce_kpi_rows(base_kpis):
    assert len(EVAL_SEEDS) == 50
    counts = base_kpis.groupby("policy").size()
    assert set(counts.index) == {"random", "nearest_job_first", "urgency_first"}
    assert (counts == 50).all()
    assert KPI_KEYS <= set(base_kpis.columns)
    assert base_kpis[list(KPI_KEYS - {"policy", "scenario"})].notna().all().all()
    assert ((base_kpis.urgent_service_rate >= 0) & (base_kpis.urgent_service_rate <= 1)).all()


def test_kpi_reward_columns_add_up_to_return(base_kpis):
    total = base_kpis[[f"r_{k}" for k in REWARD_COMPONENTS]].sum(axis=1)
    assert np.allclose(total, base_kpis["return"])


def test_heuristics_beat_random_and_mostly_end_at_the_depot(base_kpis):
    mean = base_kpis.groupby("policy").mean(numeric_only=True)
    for name in ("nearest_job_first", "urgency_first"):
        assert mean.loc[name, "return"] > mean.loc["random", "return"] + 20
        assert mean.loc[name, "overflow_bin_minutes"] < mean.loc["random", "overflow_bin_minutes"]
        # The end-of-shift rule brings most trucks home empty (breakdowns can still strand one).
        assert mean.loc[name, "vehicles_not_returned"] < 1.0
        assert mean.loc[name, "tonnes_left_on_trucks"] < 1.0


def test_urgency_first_serves_more_urgent_requests(base_kpis):
    rate = base_kpis.groupby("policy")["urgent_service_rate"].mean()
    assert rate["urgency_first"] > rate["nearest_job_first"] + 0.2
    assert rate["urgency_first"] > rate["random"]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_policy_runs_in_every_scenario(scenario):
    for cls in HEURISTICS.values():
        rows = run_policy(MaskCheckingPolicy(cls()), scenario, seeds=range(3))
        assert len(rows) == 3 and all(r["scenario"] == scenario for r in rows)


def test_policies_are_deterministic_per_seed():
    env = WasteFlowEnv()
    for cls in (RandomPolicy, NearestJobFirstPolicy, UrgencyFirstPolicy):
        assert run_episode(env, cls(), 5) == run_episode(env, cls(), 5)
    # The random policy's own stream depends on the seed too.
    assert run_episode(env, RandomPolicy(), 5)["return"] != run_episode(env, RandomPolicy(), 6)["return"]


def test_nearest_job_first_picks_nearest_bin_above_threshold():
    env = WasteFlowEnv()
    obs, _ = env.reset(seed=0)
    sim = env.sim
    sim.fill[:] = 0.1
    node = int(sim.vehicle_node[sim.deciding_vehicle])
    order = np.argsort(sim.layout.dist_km[node, :15])
    near, far = int(order[0]), int(order[-1])
    sim.fill[[near, far]] = 0.9
    obs = env._obs()
    assert NearestJobFirstPolicy().act(obs, env) == near
    sim.fill[near] = 0.2
    assert NearestJobFirstPolicy().act(env._obs(), env) == far


def test_urgency_first_prefers_earliest_reachable_deadline_then_fullest():
    env = WasteFlowEnv()
    env.reset(seed=0)
    sim = env.sim
    sim.fill[:] = 0.4
    sim.fill[3] = 0.95
    sim.urgent_deadline[[7, 9]] = [sim.t + 80.0, sim.t + 60.0]
    assert UrgencyFirstPolicy().act(env._obs(), env) == 9
    sim.urgent_deadline[[7, 9]] = np.inf
    assert UrgencyFirstPolicy().act(env._obs(), env) == 3


def test_dispatchers_head_home_near_shift_end():
    env = WasteFlowEnv()
    env.reset(seed=0)
    sim = env.sim
    v = sim.deciding_vehicle
    sim.vehicle_node[v] = 0
    sim.vehicle_load_kg[v] = 500.0
    sim.fill[:] = 0.9
    sim.t = sim.config.shift_minutes - 25.0
    for policy in (NearestJobFirstPolicy(), UrgencyFirstPolicy()):
        assert policy.act(env._obs(), env) == 15  # disposal, which ends at the depot
