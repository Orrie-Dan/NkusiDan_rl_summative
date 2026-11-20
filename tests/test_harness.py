"""Card 16: evaluation harness (fixed seeds, extra KPIs, bootstrap CIs, paired differences).

Done-when (a KPI table for all baselines) is the CLI run ``uv run main.py evaluate``; these tests use few seeds.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from baselines.heuristics import NearestJobFirstPolicy, RandomPolicy, UrgencyFirstPolicy
from environment.custom_env import REWARD_COMPONENTS
from environment.simulation import ACTION_WAIT, VehicleStatus
from evaluation import harness
from evaluation.harness import (
    KPIS, bootstrap_ci, evaluate, evaluate_policy, extra_kpis, make_policy, paired_differences, summarize,
)
from evaluation.rollout import EVAL_SEEDS

SEEDS = EVAL_SEEDS[:4]


class AlwaysWait:
    name = "always_wait"

    def reset(self, seed=None):
        pass

    def act(self, obs, env):
        assert obs["action_mask"][ACTION_WAIT]  # an empty truck can always wait
        return ACTION_WAIT


@pytest.fixture(scope="module")
def urgency_rows():
    return pd.DataFrame(evaluate_policy(UrgencyFirstPolicy(), "base", SEEDS))


def test_rows_have_every_kpi_and_components_sum_to_return(urgency_rows):
    assert set(KPIS) <= set(urgency_rows.columns)
    total = urgency_rows[[f"r_{k}" for k in REWARD_COMPONENTS]].sum(axis=1)
    np.testing.assert_allclose(total, urgency_rows["return"], atol=1e-6)
    assert urgency_rows["vehicle_utilization"].between(0, 1).all()
    assert urgency_rows["bins_full_at_end"].between(0, 15).all()
    assert (urgency_rows["avg_disposal_load_t"] > 0).all()


def test_evaluation_is_deterministic_and_matches_across_policy_instances(urgency_rows):
    again = pd.DataFrame(evaluate_policy(UrgencyFirstPolicy(), "base", SEEDS))
    pd.testing.assert_frame_equal(urgency_rows, again)


def test_idle_fleet_has_zero_utilization_and_full_bins():
    rows = evaluate_policy(AlwaysWait(), "base", SEEDS[:2])
    for r in rows:
        assert r["vehicle_utilization"] == 0.0
        assert r["tonnes_collected"] == 0.0 and r["disposal_trips"] == 0
        assert np.isnan(r["avg_disposal_load_t"])
        assert r["bins_full_at_end"] > 0  # 1.25x demand fills some bins in 8 h


def _stub_env(decisions_shift=100.0, out_of_service=(False, False, False), route_arrive_at=(0, 0, 0),
              status=(VehicleStatus.IDLE,) * 3, kg_disposed=0.0, disposal_trips=0, fill=(0.5,) * 15):
    cfg = SimpleNamespace(shift_minutes=decisions_shift, n_vehicles=3)
    sim = SimpleNamespace(
        config=cfg, vehicle_out_of_service=np.array(out_of_service), route_arrive_at=np.array(route_arrive_at, float),
        vehicle_status=np.array(status), fill=np.array(fill),
        stats=SimpleNamespace(kg_disposed=kg_disposed, disposal_trips=disposal_trips),
    )
    return SimpleNamespace(sim=sim)


def test_utilization_counts_busy_intervals_until_next_decision_or_shift_end():
    # v0: busy 0-40, waits 40-60, busy 60-100. v1: busy the whole shift. v2: waits the whole shift.
    decisions = [(0, 0.0, 3), (1, 0.0, 5), (2, 0.0, ACTION_WAIT), (0, 40.0, ACTION_WAIT), (0, 60.0, 15)]
    k = extra_kpis(_stub_env(), decisions)
    assert k["vehicle_utilization"] == pytest.approx((40 + 40 + 100) / 300)


def test_permanent_breakdown_stops_the_busy_clock():
    decisions = [(0, 0.0, 3), (1, 0.0, ACTION_WAIT), (2, 0.0, ACTION_WAIT)]
    env = _stub_env(out_of_service=(True, False, False), route_arrive_at=(25.0, 0, 0))
    assert extra_kpis(env, decisions)["vehicle_utilization"] == pytest.approx(25 / 300)


def test_avg_disposal_load_skips_a_disposal_in_progress_and_counts_full_bins():
    env = _stub_env(status=(VehicleStatus.DISPOSING, VehicleStatus.IDLE, VehicleStatus.IDLE),
                    kg_disposed=6_000.0, disposal_trips=4, fill=(1.0, 1.0) + (0.3,) * 13)
    k = extra_kpis(env, [])
    assert k["avg_disposal_load_t"] == pytest.approx(6.0 / 3)  # 3 completed disposals
    assert k["bins_full_at_end"] == 2


def test_bootstrap_ci():
    assert bootstrap_ci([2.0] * 10) == (2.0, 2.0, 2.0)
    assert all(np.isnan(bootstrap_ci([np.nan, np.nan])))
    x = np.random.default_rng(1).normal(5.0, 1.0, 200)
    mean, lo, hi = bootstrap_ci(x)
    assert lo < mean < hi and hi - lo < 0.5
    assert bootstrap_ci(x) == (mean, lo, hi)  # fixed bootstrap seed
    assert bootstrap_ci([1.0, np.nan, 3.0])[0] == 2.0


def _episodes(values: dict[str, list[float]]) -> pd.DataFrame:
    rows = [{"policy": p, "scenario": "base", "seed": i, "return": v, **{k: v for k in harness.PAIRED_KPIS[1:]}}
            for p, vals in values.items() for i, v in enumerate(vals)]
    return pd.DataFrame(rows)


def test_paired_differences_pair_on_seed():
    eps = _episodes({"ref": [10.0, 20.0, 30.0, 40.0], "better": [11.0, 21.0, 31.0, 41.0],
                     "same": [10.0, 20.0, 30.0, 40.0]})
    pd_ = paired_differences(eps, "ref").query("kpi == 'return'").set_index("policy")
    # a constant +1 per seed is significant even though the unpaired spread is large
    assert pd_.loc["better", ["mean_diff", "ci_low", "ci_high"]].tolist() == [1.0, 1.0, 1.0]
    assert pd_.loc["better", "significant"] and not pd_.loc["same", "significant"]
    assert set(pd_.index) == {"better", "same"} and (pd_["n"] == 4).all()


def test_summarize_long_format():
    s = summarize(_episodes({"a": [1.0, 3.0], "b": [5.0, 5.0]}), kpis=["return"])
    assert s[["policy", "kpi", "mean", "n"]].values.tolist() == [["a", "return", 2.0, 2], ["b", "return", 5.0, 2]]


def test_evaluate_writes_csvs_and_rejects_bad_input(tmp_path):
    res = evaluate(["random", "urgency_first"], ["base", "heavy_traffic"], SEEDS[:2], out=tmp_path)
    assert len(res.episodes) == 2 * 2 * 2
    assert {"episodes.csv", "kpi_summary.csv", "paired_vs_urgency_first.csv"} <= {p.name for p in tmp_path.iterdir()}
    assert set(res.paired["policy"]) == {"random"} and set(res.paired["scenario"]) == {"base", "heavy_traffic"}
    with pytest.raises(ValueError):
        evaluate(["random", "random"], seeds=SEEDS[:1])
    with pytest.raises(ValueError):
        evaluate(["random"], ["nowhere"], seeds=SEEDS[:1])
    with pytest.raises(ValueError):
        make_policy("no_such_policy")


def test_make_policy_baselines_and_checkpoint(tmp_path):
    assert isinstance(make_policy("random"), RandomPolicy)
    assert isinstance(make_policy("nearest_job_first"), NearestJobFirstPolicy)
    from training.common import RunConfig, mlp, save_checkpoint
    from environment.simulation import N_ACTIONS

    run_dir = tmp_path / "ppo_x_s0"
    run_dir.mkdir()
    save_checkpoint(run_dir / "model.pt", mlp(N_ACTIONS, [8]), RunConfig(algo="ppo", hidden_sizes=[8]), 0, None)
    assert make_policy(str(run_dir / "model.pt")).name == "ppo_x_s0"
    pol = make_policy(f"mine={run_dir / 'model.pt'}")
    assert pol.name == "mine" and pol.mode == "sample"
    rows = evaluate_policy(pol, "base", SEEDS[:1])
    assert rows[0]["policy"] == "mine"
