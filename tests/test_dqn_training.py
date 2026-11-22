"""Card 10: DQN trainer.

Done-when ("a short run learns better than random on the base scenario") is checked by a manual 100k-frame run,
too slow for the suite. These tests cover the pieces that run must rely on.
"""

import csv
import json

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import QValueActor
from torchrl.objectives import DQNLoss

from environment.custom_env import OBS_DIM
from environment.custom_env import N_ACTIONS
from training import load_policy, make_env
from training.dqn_training import METRIC_COLUMNS, DQNConfig, _parse, train


def _tiny(**kw) -> DQNConfig:
    base = dict(name="tiny", total_frames=1_200, frames_per_batch=200, learning_starts=400, updates_per_batch=5,
                batch_size=32, buffer_size=1_000, eps_decay_frames=1_000, hidden_sizes=[16],
                eval_every_frames=600, eval_episodes=1)
    return DQNConfig(**{**base, **kw})


@pytest.mark.parametrize("variant", [dict(), dict(target_update="hard", hard_update_interval=3, double_dqn=True)])
def test_tiny_run_trains_logs_and_checkpoints(tmp_path, variant):
    cfg = _tiny(**variant)
    summary = train(cfg, tmp_path)
    log_dir, model_dir = tmp_path / "logs" / "dqn" / "tiny_s0", tmp_path / "models" / "dqn" / "tiny_s0"
    assert {p.name for p in model_dir.iterdir()} == {"model.pt", "model_final.pt", "config.json"}
    assert DQNConfig.load(model_dir / "config.json") == cfg

    rows = list(csv.DictReader(open(log_dir / "train_log.csv", encoding="utf-8")))
    assert list(rows[0])[-len(METRIC_COLUMNS):] == METRIC_COLUMNS
    train_rows = [r for r in rows if r["updates"] != ""]
    assert [int(r["frames"]) for r in train_rows] == [200, 400, 600, 800, 1000, 1200]
    # no gradient steps before learning_starts, then updates_per_batch per batch
    assert [int(r["updates"]) for r in train_rows] == [0, 5, 10, 15, 20, 25]
    assert all(r["loss"] == "" for r in train_rows[:1]) and all(float(r["loss"]) >= 0 for r in train_rows[1:])
    eps = [float(r["epsilon"]) for r in train_rows]
    assert eps[0] == 1.0 and all(a > b for a, b in zip(eps, eps[1:]))
    assert summary["updates"] == 25 and json.loads((log_dir / "summary.json").read_text())["updates"] == 25

    policy = load_policy(model_dir / "model_final.pt")
    assert policy.name == "dqn"


def test_epsilon_stops_at_eps_end(tmp_path):
    train(_tiny(eps_decay_frames=400, eps_end=0.1, updates_per_batch=1), tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "logs" / "dqn" / "tiny_s0" / "train_log.csv", encoding="utf-8")))
    eps = [float(r["epsilon"]) for r in rows if r["updates"] != ""]
    assert eps[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("double_dqn", [False, True])
def test_invalid_next_actions_never_enter_the_target(double_dqn):
    """Action 0 has Q=100 everywhere but is masked in s'. A correct target is r + gamma * max over valid = 0."""
    env = make_env()
    net = nn.Linear(OBS_DIM, N_ACTIONS)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.zero_()
        net.bias[0] = 100.0
    actor = QValueActor(net, in_keys=["observation"], spec=env.action_spec, action_mask_key="action_mask")
    loss = DQNLoss(actor, action_space=env.action_spec, delay_value=True, double_dqn=double_dqn, loss_function="l2")
    loss.make_value_estimator(gamma=1.0)

    n = 4
    mask = torch.ones(n, N_ACTIONS, dtype=torch.bool)
    next_mask = mask.clone()
    next_mask[:, 0] = False
    td = TensorDict(
        {
            "observation": torch.rand(n, OBS_DIM),
            "action_mask": mask,
            "action": torch.full((n,), 1),  # taken action: Q = 0
            "next": {
                "observation": torch.rand(n, OBS_DIM),
                "action_mask": next_mask,
                "reward": torch.zeros(n, 1),
                "done": torch.zeros(n, 1, dtype=torch.bool),
                "terminated": torch.zeros(n, 1, dtype=torch.bool),
            },
        },
        batch_size=[n],
    )
    assert loss(td)["loss"].item() == pytest.approx(0.0, abs=1e-8)
    next_mask[:, 0] = True  # sanity: unmasked, the 100 does leak in
    assert loss(td)["loss"].item() > 1_000


def test_config_validation():
    for bad in (dict(target_update="polyak"), dict(loss_function="huber"), dict(eps_start=0.1, eps_end=0.5),
                dict(tau=0.0), dict(batch_size=10, buffer_size=5), dict(algo="ppo")):
        with pytest.raises(ValueError):
            DQNConfig(**bad)


def test_cli_merges_config_file_and_flag_overrides(tmp_path):
    path = tmp_path / "c.json"
    DQNConfig(name="from_file", lr=1e-3, buffer_size=7_000).save(path)
    cfg, root = _parse(["--config", str(path), "--lr", "0.002", "--double-dqn", "true",
                        "--hidden-sizes", "64,32", "--root", "out"])
    assert (cfg.name, cfg.lr, cfg.buffer_size, cfg.double_dqn, cfg.hidden_sizes, root) == \
        ("from_file", 0.002, 7_000, True, [64, 32], "out")
    assert _parse([])[0] == DQNConfig()
