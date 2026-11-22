"""Card 03: event-driven simulation core.

Done-when check: a scripted policy runs full 8-hour shifts in every scenario and the episode stats are plausible.
Run with ``uv run pytest tests/test_simulation.py -s`` to print the per-scenario stats table.
"""

import numpy as np
import pytest

from environment.custom_env import (
    ACTION_DISPOSAL,
    ACTION_WAIT,
    FACILITY,
    N_ACTIONS,
    SCENARIOS,
    ScenarioConfig,
    VehicleStatus,
    WasteFlowSim,
    make_sim,
)


def scripted_policy(sim: WasteFlowSim) -> int:
    """Simple greedy rule used only to exercise the simulation: urgent first, then fullest nearby bin."""
    v = sim.deciding_vehicle
    mask = sim.action_mask()
    cap = sim.config.vehicle_capacity_kg
    load_frac = sim.vehicle_load_kg[v] / cap
    if load_frac >= 0.85 and mask[ACTION_DISPOSAL]:
        return ACTION_DISPOSAL
    locs = np.flatnonzero(mask[:FACILITY])
    dist = sim.layout.dist_km[sim.vehicle_node[v], locs]
    worthwhile = (sim.fill[locs] >= 0.35) | sim.urgent_pending[locs]
    if worthwhile.any():
        score = 2.0 * sim.urgent_pending[locs] + sim.fill[locs] - 0.03 * dist
        score[~worthwhile] = -np.inf
        return int(locs[np.argmax(score)])
    if mask[ACTION_DISPOSAL] and load_frac > 0.3:
        return ACTION_DISPOSAL
    return ACTION_WAIT if mask[ACTION_WAIT] else ACTION_DISPOSAL


def run_episode(sim: WasteFlowSim, seed: int, policy=scripted_policy, check_invariants: bool = False):
    sim.reset(seed=seed)
    outcomes = []
    last_t = sim.t
    while not sim.done:
        mask = sim.action_mask()
        assert mask.any(), "mask must always leave a valid action"
        if check_invariants:
            v = sim.deciding_vehicle
            assert sim.vehicle_status[v] == VehicleStatus.IDLE
            assert not sim.vehicle_out_of_service[v]
            # Mask matches state: disposal only with a load, waiting only with spare capacity,
            # and no bin that another vehicle is already heading to.
            load, cap = sim.vehicle_load_kg[v], sim.config.vehicle_capacity_kg
            assert mask[ACTION_DISPOSAL] == (load > 1e-9)
            assert mask[ACTION_WAIT] == (load < cap - 1e-9)
            others = sim.vehicle_target[np.arange(sim.config.n_vehicles) != v]
            assert not mask[others[(others >= 0) & (others < FACILITY)]].any()
        out = sim.apply_action(policy(sim))
        outcomes.append(out)
        if check_invariants:
            assert sim.t >= last_t
            # No two vehicles are ever heading to the same bin.
            bins = sim.vehicle_target[(sim.vehicle_target >= 0) & (sim.vehicle_target < FACILITY)]
            assert len(bins) == len(np.unique(bins))
            assert np.all((sim.fill >= 0) & (sim.fill <= 1))
            assert np.all(sim.vehicle_load_kg >= -1e-9)
            assert np.all(sim.vehicle_load_kg <= sim.config.vehicle_capacity_kg + 1e-6)
            assert out.travel_minutes >= 0 and out.overflow_bin_minutes >= 0 and out.kg_collected >= 0
            last_t = sim.t
    return outcomes


def random_policy(rng):
    def policy(sim):
        return int(rng.choice(np.flatnonzero(sim.action_mask())))

    return policy


# ---------------------------------------------------------------------------------------------- done-when


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scripted_policy_runs_full_shift_with_plausible_stats(scenario):
    sim = make_sim(scenario)
    rows = []
    for seed in range(10):
        outcomes = run_episode(sim, seed, check_invariants=True)
        s = sim.stats
        assert sim.t == pytest.approx(sim.config.shift_minutes)
        assert outcomes[-1].done and not any(o.done for o in outcomes[:-1])
        assert sim.deciding_vehicle is None
        # Step outcomes add up to the episode totals.
        assert sum(o.kg_collected for o in outcomes) == pytest.approx(s.kg_collected)
        assert sum(o.travel_minutes for o in outcomes) == pytest.approx(s.travel_minutes)
        assert sum(o.overflow_bin_minutes for o in outcomes) == pytest.approx(s.overflow_bin_minutes)
        assert sum(o.urgent_missed for o in outcomes) == s.urgent_missed
        # Mass balance: everything collected is either disposed of or still on a truck.
        assert s.kg_collected == pytest.approx(s.kg_disposed + s.load_remaining_kg)
        assert s.urgent_served + s.urgent_missed <= s.urgent_arrived
        # Plausibility: real work happens, but not more than a 3-truck fleet can do in 8 hours.
        assert 20 <= s.decisions <= 400
        assert s.bin_visits >= 15 and s.disposal_trips >= 3
        assert 3_000 < s.kg_collected < 40_000
        assert 0 < s.travel_minutes < 3 * sim.config.shift_minutes
        assert 0 <= s.overflow_bin_minutes <= 15 * sim.config.shift_minutes
        rows.append(s)

    def mean(attr):
        return np.mean([getattr(r, attr) for r in rows])

    print(
        f"\n[{scenario:>14}] 10 episodes, scripted policy | "
        f"decisions {mean('decisions'):.0f} | visits {mean('bin_visits'):.0f} | disposals {mean('disposal_trips'):.1f} | "
        f"collected {mean('kg_collected'):.0f} kg | travel {mean('travel_minutes'):.0f} min | "
        f"overflow {mean('overflow_bin_minutes'):.0f} bin-min | urgent {mean('urgent_served'):.1f}/"
        f"{mean('urgent_arrived'):.1f} served, {mean('urgent_missed'):.1f} missed | "
        f"breakdowns {mean('breakdowns'):.2f} | left on trucks {mean('load_remaining_kg'):.0f} kg | "
        f"not returned {mean('vehicles_not_returned'):.1f}"
    )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_random_masked_policy_never_errors(scenario):
    sim = make_sim(scenario)
    rng = np.random.default_rng(123)
    for seed in range(20):
        run_episode(sim, seed, policy=random_policy(rng), check_invariants=True)
        assert sim.done


# ---------------------------------------------------------------------------------------------- determinism


def test_same_seed_same_trajectory():
    a, b = make_sim(), make_sim()
    oa, ob = run_episode(a, 7), run_episode(b, 7)
    assert [(o.vehicle, o.action, o.end_time, o.kg_collected) for o in oa] == [
        (o.vehicle, o.action, o.end_time, o.kg_collected) for o in ob
    ]
    assert a.stats == b.stats


def test_reset_reuses_instance_deterministically():
    sim = make_sim()
    run_episode(sim, 3)
    first = sim.stats
    run_episode(sim, 99)
    run_episode(sim, 3)
    assert sim.stats == first


def test_different_seeds_differ_but_layout_is_fixed():
    sim = make_sim()
    sim.reset(seed=1)
    fill1, xy1 = sim.fill.copy(), sim.layout.node_xy.copy()
    sim.reset(seed=2)
    assert not np.allclose(fill1, sim.fill)
    assert np.array_equal(xy1, make_sim().layout.node_xy)


# ---------------------------------------------------------------------------------------------- mechanics


def test_reset_state():
    sim = make_sim()
    sim.reset(seed=0)
    assert sim.t == 0 and sim.deciding_vehicle == 0
    assert np.all(sim.vehicle_node == FACILITY)
    assert np.all(sim.vehicle_load_kg == 0)
    mask = sim.action_mask()
    assert mask.shape == (N_ACTIONS,) and mask.dtype == bool
    assert not mask[ACTION_DISPOSAL] and mask[ACTION_WAIT]  # empty truck: cannot dispose, may wait


def test_simultaneous_free_vehicles_decide_without_time_passing():
    sim = make_sim(breakdown_prob_per_trip=0.0)
    sim.reset(seed=0)
    o0 = sim.apply_action(0)
    assert o0.end_time == 0 and sim.deciding_vehicle == 1
    o1 = sim.apply_action(1)
    assert o1.end_time == 0 and sim.deciding_vehicle == 2
    sim.apply_action(2)
    assert sim.t > 0


def test_mask_blocks_bins_targeted_by_other_vehicles():
    sim = make_sim(breakdown_prob_per_trip=0.0)
    sim.reset(seed=0)
    sim.apply_action(4)
    assert not sim.action_mask()[4]
    with pytest.raises(ValueError):
        sim.apply_action(4)


def test_mask_blocks_bins_that_do_not_fit_and_wait_when_full():
    sim = make_sim()
    sim.reset(seed=0)
    v = sim.deciding_vehicle
    cap = sim.config.vehicle_capacity_kg
    sim.vehicle_load_kg[v] = cap - 100.0
    mask = sim.action_mask()
    bin_kg = sim.fill * sim.layout.bin_capacity_kg
    assert np.array_equal(mask[:FACILITY], bin_kg <= 100.0)
    sim.vehicle_load_kg[v] = cap
    mask = sim.action_mask()
    assert not mask[ACTION_WAIT] and mask[ACTION_DISPOSAL]


def test_collect_then_dispose_moves_load_through_the_system():
    sim = make_sim(breakdown_prob_per_trip=0.0, urgent_rate_per_hour=0.0)
    sim.reset(seed=0)
    target = int(np.argmax(sim.fill))
    out = sim.apply_action(target)  # vehicle 0
    sim.apply_action(ACTION_WAIT)  # vehicle 1
    sim.apply_action(ACTION_WAIT)  # vehicle 2
    while sim.deciding_vehicle != 0:
        sim.apply_action(ACTION_WAIT)
    load = sim.vehicle_load_kg[0]
    assert load > 0 and sim.vehicle_node[0] == target
    assert sim.fill[target] < 0.05  # bin emptied, apart from fill accrued during service
    assert out.travel_minutes > 0
    sim.apply_action(ACTION_DISPOSAL)
    while sim.deciding_vehicle != 0:
        sim.apply_action(ACTION_WAIT)
    assert sim.vehicle_load_kg[0] == 0 and sim.stats.kg_disposed == pytest.approx(load)


def test_wait_lasts_10_to_15_minutes():
    sim = make_sim(urgent_rate_per_hour=0.0)
    sim.reset(seed=0)
    for _ in range(3):
        sim.apply_action(ACTION_WAIT)
    assert 10.0 <= sim.t <= 15.0


def test_overflow_accrues_when_nobody_collects():
    sim = make_sim(demand_multiplier=3.0, urgent_rate_per_hour=0.0)
    sim.reset(seed=0)
    while not sim.done:
        sim.apply_action(ACTION_WAIT)
    assert np.all(sim.fill == 1.0)
    assert sim.stats.overflow_bin_minutes > 0
    assert sim.stats.kg_collected == 0 and sim.stats.vehicles_not_returned == 0


def test_urgent_requests_get_missed_without_service():
    sim = make_sim(urgent_rate_per_hour=6.0)
    sim.reset(seed=0)
    while not sim.done:
        sim.apply_action(ACTION_WAIT)
    s = sim.stats
    assert s.urgent_arrived > 10
    assert s.urgent_served == 0
    assert s.urgent_missed >= s.urgent_arrived - 15  # only requests still pending at shift end are not missed


def test_permanent_breakdown_takes_vehicle_out_and_releases_its_bin():
    sim = make_sim(breakdown_prob_per_trip=1.0, breakdown_rest_of_shift_prob=1.0, urgent_rate_per_hour=0.0)
    sim.reset(seed=0)
    sim.apply_action(3)  # vehicle 0 will break down on the way
    sim.apply_action(ACTION_WAIT)
    sim.apply_action(ACTION_WAIT)
    while not sim.vehicle_out_of_service[0]:
        sim.apply_action(ACTION_WAIT)
    assert sim.vehicle_status[0] == VehicleStatus.BROKEN_DOWN
    assert sim.vehicle_target[0] == -1
    while not sim.done:
        assert sim.deciding_vehicle != 0  # never asked to decide again
        sim.apply_action(ACTION_WAIT)
    assert sim.stats.breakdowns == 1 and sim.stats.vehicles_not_returned == 1


def test_temporary_breakdown_delays_arrival():
    kw = dict(urgent_rate_per_hour=0.0, travel_noise_sigma=0.0, traffic_p_congest=0.0)
    fast, slow = make_sim(breakdown_prob_per_trip=0.0, **kw), make_sim(
        breakdown_prob_per_trip=1.0, breakdown_rest_of_shift_prob=0.0, **kw
    )
    free_times = []
    for sim in (fast, slow):
        sim.reset(seed=0)
        sim.apply_action(5)  # vehicle 0
        sim.apply_action(ACTION_WAIT)
        sim.apply_action(ACTION_WAIT)
        while sim.deciding_vehicle != 0:
            sim.apply_action(ACTION_WAIT)
        assert sim.vehicle_node[0] == 5
        free_times.append(sim.t)  # arrival + service time
    assert slow.stats.breakdowns == 1 and fast.stats.breakdowns == 0
    # Service time grows slightly with the extra fill accrued during the delay, so allow a few minutes of slack.
    assert free_times[1] - free_times[0] >= slow.config.breakdown_downtime_range_min[0] - 5.0


def test_traffic_congestion_slows_travel():
    calm = make_sim(traffic_p_congest=0.0, travel_noise_sigma=0.0)
    jam = make_sim(traffic_p_congest=1.0, traffic_p_clear=0.0, travel_noise_sigma=0.0)
    for sim in (calm, jam):
        sim.reset(seed=0)
        while sim.t <= sim.config.tick_minutes:  # pass the first tick so the jam scenario becomes congested
            sim.apply_action(ACTION_WAIT)
    assert jam.zone_congested.all() and not calm.zone_congested.any()
    assert jam.travel_time(FACILITY, 0) == pytest.approx(calm.travel_time(FACILITY, 0) * 1.6)


def test_actions_after_shift_end_raise():
    sim = make_sim()
    run_episode(sim, 0)
    with pytest.raises(RuntimeError):
        sim.apply_action(ACTION_WAIT)


def test_config_rejects_other_fleet_sizes():
    with pytest.raises(ValueError):
        ScenarioConfig(n_vehicles=4)
