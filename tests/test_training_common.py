"""Card 09: shared RL infrastructure.

Done-when: a dummy run writes the full model and log directories.
"""

import csv
import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn
from torchrl.envs.utils import check_env_specs

import training as common
from environment.custom_env import OBS_DIM, RewardConfig, WasteFlowEnv
from environment.custom_env import N_ACTIONS
from training import (
    BASE_LOG_COLUMNS,
    VALIDATION_SEEDS,
    RunConfig,
    TorchPolicy,
    TrainingRun,
    dummy_run,
    load_policy,
    log_dir_for,
    make_actor,
    make_critic,
    make_env,
    make_qnet,
    masked_argmax,
    mlp,
    model_dir_for,
    save_checkpoint,
    seed_everything,
)
from environment.custom_env import EVAL_SEEDS


@dataclass
class _DemoConfig(RunConfig):
    """Stand-in for an algorithm config that adds its own hyperparameters."""

    eps_start: float = 1.0
    buffer_size: int = 50_000


# ------------------------------------------------------------------------------------------------- done-when


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    return root, dummy_run(root, algo="ppo", total_frames=1_500)


def test_dummy_run_writes_full_model_and_log_directories(dummy):
    root, summary = dummy
    model_dir, log_dir = root / "models" / "pg" / "ppo_dummy_s0", root / "logs" / "ppo" / "dummy_s0"
    assert {p.name for p in model_dir.iterdir()} == {"model.pt", "model_final.pt", "config.json"}
    assert {p.name for p in log_dir.iterdir()} == {"train_log.csv", "summary.json", "config.json"}

    rows = list(csv.DictReader(open(log_dir / "train_log.csv", encoding="utf-8")))
    assert list(rows[0]) == list(BASE_LOG_COLUMNS) + ["entropy", "policy_loss", "value_loss"]
    train_rows = [r for r in rows if r["updates"] != ""]
    eval_rows = [r for r in rows if r["eval_return_mean"] != ""]
    assert [int(r["frames"]) for r in train_rows] == [500, 1000, 1500]
    assert all(float(r["entropy"]) > 0 for r in train_rows)
    assert len(eval_rows) >= 2  # periodic + final

    saved = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["frames"] == 1500 and saved["episodes"] > 0 and saved["frames_per_s"] > 0
    assert saved["best_eval_return"] == pytest.approx(max(float(r["eval_return_mean"]) for r in eval_rows), abs=1e-5)
    assert RunConfig.load(model_dir / "config.json") == RunConfig.from_dict(saved["config"])


def test_saved_best_model_loads_as_a_valid_greedy_policy(dummy):
    root, _ = dummy
    policy = load_policy(root / "models" / "pg" / "ppo_dummy_s0" / "model.pt")
    env = WasteFlowEnv()
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        a = policy.act(obs, env)
        assert obs["action_mask"][a]
        obs, _, done, _, _ = env.step(a)


# ------------------------------------------------------------------------------------------------------ config


def test_run_config_json_roundtrip_including_subclass_fields(tmp_path):
    cfg = _DemoConfig(algo="dqn", name="dqn_c03", seed=2, hidden_sizes=[64, 64, 64], eps_start=0.5)
    cfg.save(tmp_path / "c.json")
    loaded = _DemoConfig.load(tmp_path / "c.json")
    assert loaded == cfg and loaded.run_id == "dqn_c03_s2"
    assert loaded.reward_config == RewardConfig()


def test_run_config_rejects_bad_values():
    with pytest.raises(ValueError):
        RunConfig(algo="sac")
    with pytest.raises(ValueError):
        RunConfig(activation="gelu")
    with pytest.raises(ValueError):
        RunConfig(eval_episodes=len(VALIDATION_SEEDS) + 1)
    with pytest.raises(ValueError):
        RunConfig.from_dict({"algo": "dqn", "typo_field": 1})


def test_output_paths_follow_the_brief(tmp_path):
    assert model_dir_for("dqn", "c01_s0", tmp_path) == tmp_path / "models" / "dqn" / "c01_s0"
    assert model_dir_for("a2c", "c01_s0", tmp_path) == tmp_path / "models" / "pg" / "a2c_c01_s0"
    assert log_dir_for("reinforce", "c01_s0", tmp_path) == tmp_path / "logs" / "reinforce" / "c01_s0"


def test_validation_seeds_are_disjoint_from_eval_and_training_seeds():
    assert not set(VALIDATION_SEEDS) & set(EVAL_SEEDS)
    assert min(VALIDATION_SEEDS) > 1000  # training seeds are small integers


# -------------------------------------------------------------------------------------------------------- env


def test_make_env_specs_scenario_and_reward_config():
    env = make_env("heavy_traffic", seed=0, reward_config=RewardConfig(alpha_per_tonne=0.0))
    check_env_specs(env)
    inner = env.base_env._env.unwrapped
    assert inner.sim.config.name == "heavy_traffic" and inner.reward_config.alpha_per_tonne == 0.0
    td = env.rollout(150, break_when_any_done=False)
    assert td["action_mask"].dtype == torch.bool and env.action_spec.shape == ()
    acts = td["action"].reshape(-1)
    assert td["action_mask"].reshape(-1, N_ACTIONS)[torch.arange(acts.numel()), acts].all()
    eps = common.completed_episode_returns(td)
    assert len(eps) == int(td["next", "done"].sum())


def test_seed_everything_pins_threads_and_makes_init_reproducible():
    seed_everything(5)
    assert torch.get_num_threads() == 1
    a = mlp(N_ACTIONS, [16])
    seed_everything(5)
    b = mlp(N_ACTIONS, [16])
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


# --------------------------------------------------------------------------------------------------- networks


def test_builders_have_configurable_depth_width_and_activation():
    net = mlp(N_ACTIONS, [64, 32, 16], "relu")
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(OBS_DIM, 64), (64, 32), (32, 16), (16, N_ACTIONS)]
    assert sum(isinstance(m, nn.ReLU) for m in net.modules()) == 3

    env = make_env()
    actor, logits_net = make_actor(env.action_spec, [32], "tanh")
    critic = make_critic([32])
    qactor, q_net = make_qnet(env.action_spec, [32])
    td = env.reset()
    out = actor(td.clone())
    assert out["action_mask"][out["action"]]  # sampled action is valid
    assert critic(td.clone())["state_value"].shape == (1,)
    q = qactor(td.clone())
    assert q["action_mask"][q["action"]]
    assert logits_net(td["observation"]).shape == q_net(td["observation"]).shape == (N_ACTIONS,)


# ---------------------------------------------------------------------------------------------- masked actions


def test_masked_argmax_respects_the_mask():
    scores = torch.tensor([[5.0, 1.0, 3.0], [0.0, 9.0, -1.0]])
    mask = torch.tensor([[False, True, True], [True, False, True]])
    assert masked_argmax(scores, mask).tolist() == [2, 0]
    with pytest.raises(ValueError):
        masked_argmax(scores, torch.tensor([[False, False, False], [True, True, True]]))


def test_torch_policy_never_picks_masked_actions_even_when_they_score_highest():
    class Adversarial(nn.Module):  # always scores bin 0 highest
        def forward(self, x):
            s = torch.zeros(x.shape[0], N_ACTIONS)
            s[:, 0] = 100.0
            return s

    rows = common.greedy_evaluate(Adversarial(), "base", seeds=[0, 1])
    assert len(rows) == 2 and all(np.isfinite(r["return"]) for r in rows)


def test_checkpoint_roundtrip_gives_identical_greedy_actions(tmp_path):
    seed_everything(0)
    net = mlp(N_ACTIONS, [32, 32], "elu")
    cfg = RunConfig(algo="a2c", hidden_sizes=[32, 32], activation="elu")
    save_checkpoint(tmp_path / "m.pt", net, cfg, frames=10, eval_return=1.0)
    loaded = load_policy(tmp_path / "m.pt", mode="greedy")  # a2c checkpoints default to sampling
    original = TorchPolicy(net)
    env = WasteFlowEnv()
    obs, _ = env.reset(seed=3)
    for _ in range(30):
        a = original.act(obs, env)
        assert loaded.act(obs, env) == a
        obs, _, done, _, _ = env.step(a)
        if done:
            break
    assert loaded.name == "a2c"


def test_sampled_policy_is_valid_reproducible_per_seed_and_stochastic():
    class Flat(nn.Module):  # equal scores everywhere, invalid bin 0 boosted
        def forward(self, x):
            s = torch.zeros(x.shape[0], N_ACTIONS)
            s[:, 0] = 100.0
            return s

    def actions(policy, seed, steps=40):
        env = WasteFlowEnv()
        policy.reset(seed)
        obs, _ = env.reset(seed=seed)
        out = []
        for _ in range(steps):
            a = policy.act(obs, env)
            assert obs["action_mask"][a], "sampled an invalid action"
            out.append(a)
            obs, _, done, _, _ = env.step(a)
            if done:
                break
        return out

    policy = TorchPolicy(Flat(), mode="sample")
    assert actions(policy, 7) == actions(policy, 7)  # same seed -> same episode
    assert actions(policy, 7) != actions(policy, 8)
    assert len(set(actions(policy, 7))) > 1  # not deterministic over steps
    with pytest.raises(ValueError):
        TorchPolicy(Flat(), mode="epsilon")


def test_sampled_policy_follows_the_softmax():
    net = nn.Linear(OBS_DIM, N_ACTIONS)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.zero_()
        net.bias[1] = float(np.log(3.0))  # among valid {1, 2}: P(1) = 0.75
    policy = TorchPolicy(net, mode="sample")
    policy.reset(0)
    mask = np.zeros(N_ACTIONS, dtype=np.int8)
    mask[[1, 2]] = 1
    obs = {"observation": np.zeros(OBS_DIM, dtype=np.float32), "action_mask": mask}
    draws = [policy.act(obs) for _ in range(4000)]
    assert set(draws) == {1, 2}
    assert abs(draws.count(1) / len(draws) - 0.75) < 0.03


def test_checkpoint_eval_mode_follows_the_algorithm(tmp_path):
    net = mlp(N_ACTIONS, [8])
    for algo, mode in (("dqn", "greedy"), ("reinforce", "sample"), ("ppo", "sample")):
        save_checkpoint(tmp_path / f"{algo}.pt", net, RunConfig(algo=algo, hidden_sizes=[8]), 0, None)
        assert load_policy(tmp_path / f"{algo}.pt").mode == mode
    assert load_policy(tmp_path / "ppo.pt", mode="greedy").mode == "greedy"  # explicit override
    old = torch.load(tmp_path / "ppo.pt", weights_only=True)
    del old["eval_mode"]  # checkpoint written before eval modes existed
    torch.save(old, tmp_path / "old.pt")
    assert load_policy(tmp_path / "old.pt").mode == "greedy"


def test_sampled_runs_select_on_sampled_return_and_also_log_greedy(tmp_path, monkeypatch):
    calls = []

    def fake(net, scenario, seeds, rc=None, mode="greedy"):
        calls.append(mode)
        return [{"return": 10.0 if mode == "sample" else -5.0, "tonnes_collected": 1.0,
                 "overflow_bin_minutes": 0.0, "urgent_service_rate": 1.0}]

    monkeypatch.setattr(common, "evaluate_net", fake)
    run = TrainingRun(RunConfig(algo="reinforce", name="t", eval_episodes=1), metric_columns=[], root=tmp_path)
    result = run.maybe_evaluate(0, mlp(N_ACTIONS, [8]))
    assert calls == ["sample", "greedy"]
    assert result["eval_return_mean"] == 10.0 and result["eval_greedy_return_mean"] == -5.0
    assert run.best_eval_return == 10.0
    assert torch.load(run.model_dir / "model.pt", weights_only=True)["eval_mode"] == "sample"

    calls.clear()
    dqn = TrainingRun(RunConfig(algo="dqn", name="t", eval_episodes=1), metric_columns=[], root=tmp_path)
    assert dqn.maybe_evaluate(0, mlp(N_ACTIONS, [8]))["eval_greedy_return_mean"] == -5.0
    assert calls == ["greedy"]  # greedy runs evaluate once


# ----------------------------------------------------------------------------------------------- training run


def test_training_run_keeps_the_best_checkpoint_and_rejects_unknown_metrics(tmp_path, monkeypatch):
    returns = iter([5.0, 2.0, 9.0, 1.0])
    monkeypatch.setattr(common, "evaluate_net",
                        lambda net, scenario, seeds, rc=None, mode="greedy": [
                            {"return": next(returns), "tonnes_collected": 1.0, "overflow_bin_minutes": 0.0,
                             "urgent_service_rate": 1.0}])
    cfg = RunConfig(algo="dqn", name="t", eval_every_frames=100, eval_episodes=1)
    run = TrainingRun(cfg, metric_columns=["loss"], root=tmp_path)
    with pytest.raises(ValueError):
        run.log(10, bogus=1.0)
    with pytest.raises(ValueError):
        TrainingRun(cfg, metric_columns=["frames"], root=tmp_path)

    net = mlp(N_ACTIONS, cfg.hidden_sizes, cfg.activation)
    assert run.maybe_evaluate(0, net)["eval_return_mean"] == 5.0
    assert run.maybe_evaluate(50, net) is None  # not due yet
    run.maybe_evaluate(100, net)  # 2.0: worse, model.pt keeps frames=0
    assert torch.load(run.model_dir / "model.pt", weights_only=True)["frames"] == 0
    run.maybe_evaluate(200, net)  # 9.0: new best
    run.log(250, updates=3, episode_returns=[1.0, 3.0], loss=torch.tensor([0.5, 1.5]))
    summary = run.finish(net, 300)  # forced final eval: 1.0
    assert summary["best_eval_return"] == 9.0 and summary["best_eval_frames"] == 200
    assert summary["final_eval"]["eval_return_mean"] == 1.0
    assert torch.load(run.model_dir / "model.pt", weights_only=True)["frames"] == 200
    assert torch.load(run.model_dir / "model_final.pt", weights_only=True)["frames"] == 300
    rows = list(csv.DictReader(open(run.log_dir / "train_log.csv", encoding="utf-8")))
    logged = [r for r in rows if r["loss"] != ""]
    assert logged[0]["train_return_mean"] == "2.0" and float(logged[0]["loss"]) == 1.0
