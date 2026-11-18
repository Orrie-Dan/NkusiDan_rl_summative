"""Card 04: Gymnasium env (observation from OBS_SPEC, action mask, reward, verbose log).

Done-when: gymnasium's check_env passes and a random masked agent completes 100 episodes with no errors.
Also checks the env works through TorchRL with the card 02 transform recipe. Broader invariants are card 06.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (
    ENV_ID,
    OBS_DIM,
    OBS_SPEC,
    REWARD_COMPONENTS,
    RewardConfig,
    WasteFlowEnv,
    obs_index,
    obs_spec_table,
)
from environment.simulation import ACTION_DISPOSAL, ACTION_WAIT, N_ACTIONS, SCENARIOS


def random_masked(rng):
    return lambda obs: int(rng.choice(np.flatnonzero(obs["action_mask"])))


def run_episode(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    steps, total, info, done = 0, 0.0, {}, False
    while not done:
        obs, r, done, truncated, info = env.step(policy(obs))
        assert not truncated
        total += r
        steps += 1
    return steps, total, info


def test_gymnasium_check_env_passes():
    # check_env steps once with an *unseeded* action_space.sample() (it ignores the mask). At t=0 disposal is
    # masked (trucks are empty) and the env rejects masked actions, so the unseeded sample made this test fail
    # 1 run in 17. Seeding the action space makes that sample deterministic: seed 0 -> action 14 (a bin).
    env = WasteFlowEnv()
    env.action_space.seed(0)
    check_env(env, skip_render_check=True)


def test_random_masked_agent_completes_100_episodes():
    env, rng = WasteFlowEnv(), np.random.default_rng(0)
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            assert env.observation_space.contains(obs)
            assert obs["action_mask"][ACTION_WAIT] or obs["action_mask"][ACTION_DISPOSAL]
            obs, r, done, truncated, info = env.step(random_masked(rng)(obs))
            assert np.isfinite(r) and not truncated
        assert env.sim.t == env.sim.config.shift_minutes
        assert "episode_stats" in info and "episode_return" in info


def test_reward_equals_sum_of_components_and_episode_return():
    env, rng = WasteFlowEnv("high_demand"), np.random.default_rng(1)
    obs, _ = env.reset(seed=3)
    done, total = False, 0.0
    while not done:
        obs, r, done, _, info = env.step(random_masked(rng)(obs))
        comps = info["reward_components"]
        assert set(comps) == set(REWARD_COMPONENTS)
        assert all(type(v) is float for v in comps.values())
        assert r == pytest.approx(sum(comps.values()), abs=1e-12)
        total += r
    assert info["episode_return"] == pytest.approx(total)


def test_terminal_penalties_only_on_last_step():
    env, rng = WasteFlowEnv(), np.random.default_rng(2)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, _, done, _, info = env.step(random_masked(rng)(obs))
        terminal = info["reward_components"]["terminal_load"] + info["reward_components"]["terminal_not_returned"]
        if not done:
            assert terminal == 0.0


def test_idle_policy_is_punished_by_overflow():
    """Safeguard from the spec: waiting is never free."""
    env = WasteFlowEnv()
    _, ret, info = run_episode(env, 0, lambda obs: ACTION_WAIT if obs["action_mask"][ACTION_WAIT] else ACTION_DISPOSAL)
    assert info["episode_stats"]["overflow_bin_minutes"] > 0
    assert ret < -20


def test_same_seed_same_trajectory_and_different_seeds_differ():
    def trace(seed):
        env, rng = WasteFlowEnv(), np.random.default_rng(7)
        obs, _ = env.reset(seed=seed)
        out, done = [obs["observation"].copy()], False
        while not done:
            obs, r, done, _, _ = env.step(random_masked(rng)(obs))
            out.append(obs["observation"].copy())
            out.append(np.array([r]))
        return np.concatenate(out)

    a, b = trace(11), trace(11)
    assert np.array_equal(a, b)
    c = trace(12)
    assert a.shape != c.shape or not np.array_equal(a, c)


def test_masked_action_raises():
    env = WasteFlowEnv()
    obs, _ = env.reset(seed=0)
    assert obs["action_mask"][ACTION_DISPOSAL] == 0  # trucks start empty
    with pytest.raises(ValueError):
        env.step(ACTION_DISPOSAL)


def test_step_after_termination_raises():
    env = WasteFlowEnv()
    run_episode(env, 0, random_masked(np.random.default_rng(0)))
    with pytest.raises(RuntimeError):
        env.step(ACTION_WAIT)


def test_obs_spec_covers_vector_and_table_lists_every_feature():
    assert OBS_DIM == sum(f.size for f in OBS_SPEC) == 111
    idx = obs_index()
    assert list(idx) == [f.name for f in OBS_SPEC]
    assert idx[OBS_SPEC[-1].name].stop == OBS_DIM
    table = obs_spec_table()
    for f in OBS_SPEC:
        assert f"`{f.name}`" in table and f.meaning and f.encoding and f.source and f.cadence


def test_observation_reflects_deciding_vehicle_state():
    env = WasteFlowEnv()
    obs, info = env.reset(seed=0)
    idx, o, sim = obs_index(), obs["observation"], env.sim
    v = info["vehicle"]
    assert np.allclose(o[idx["bin_fill"]], sim.fill)
    assert o[idx["my_id"]][v] == 1.0
    assert o[idx["my_at_facility"]][0] == 1.0  # every truck starts at the facility
    assert o[idx["time_elapsed"]][0] == 0.0


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_scenario_runs(scenario):
    steps, ret, _ = run_episode(WasteFlowEnv(scenario), 0, random_masked(np.random.default_rng(0)))
    assert steps > 10 and np.isfinite(ret)


def test_custom_reward_weights_are_used():
    zero = RewardConfig(*(0.0,) * 6)
    _, ret, _ = run_episode(WasteFlowEnv(reward_config=zero), 0, random_masked(np.random.default_rng(0)))
    assert ret == 0.0


def test_verbose_prints_one_line_per_decision_and_summary(capsys):
    env = WasteFlowEnv(verbose=True)
    steps, _, _ = run_episode(env, 0, random_masked(np.random.default_rng(0)))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == steps + 1
    assert lines[0].startswith("t=06:00 V0 -> ") and "| r=" in lines[0] and "| load " in lines[0]
    assert lines[-1].startswith("== shift end")
    for line in lines:
        line.encode("cp1252")  # must print on a default Windows console


def test_registered_and_wraps_in_torchrl():
    from torchrl.envs import GymEnv, TransformedEnv
    from torchrl.envs.transforms import ActionMask, Compose, DTypeCastTransform
    from torchrl.envs.utils import check_env_specs

    assert ENV_ID in gym.registry
    env = TransformedEnv(
        GymEnv(ENV_ID, categorical_action_encoding=True),
        Compose(DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"]), ActionMask()),
    )
    env.set_seed(0)
    check_env_specs(env)
    td = env.rollout(300, break_when_any_done=False)
    actions, masks = td["action"], td["action_mask"]
    assert td["observation"].shape[-1] == OBS_DIM and masks.shape[-1] == N_ACTIONS
    assert masks[torch.arange(actions.numel()), actions].all()
    assert td["next", "done"].any()  # at least one full shift inside 300 steps
