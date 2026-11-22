"""Card 13: PPO trainer (TorchRL ClipPPOLoss + GAE, minibatch epochs).

Done-when ("a short run learns better than random") is checked by a manual run, too slow for the suite.
These tests cover the pieces that run relies on.
"""

import csv
import json

import pytest
import torch
from torchrl.collectors import Collector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from training import load_policy, make_actor, make_critic, make_env
from training.pg_training import PPO_METRIC_COLUMNS, PPOConfig, main, train_ppo


def _tiny(**kw) -> PPOConfig:
    base = dict(name="tiny", total_frames=300, frames_per_batch=100, num_epochs=2, minibatch_size=25,
                hidden_sizes=[16], eval_every_frames=150, eval_episodes=1)
    return PPOConfig(**{**base, **kw})


def _rows(tmp_path):
    rows = list(csv.DictReader(open(tmp_path / "logs" / "ppo" / "tiny_s0" / "train_log.csv", encoding="utf-8")))
    return rows, [r for r in rows if r["updates"] != ""]


@pytest.mark.parametrize("variant", [dict(), dict(loss_critic_type="l2", normalize_advantages=False, entropy_coef=0.0)])
def test_tiny_run_trains_logs_and_checkpoints(tmp_path, variant):
    cfg = _tiny(**variant)
    summary = train_ppo(cfg, tmp_path)

    model_dir = tmp_path / "models" / "pg" / "ppo_tiny_s0"
    assert {p.name for p in model_dir.iterdir()} == {"model.pt", "model_final.pt", "config.json"}
    assert PPOConfig.load(model_dir / "config.json") == cfg
    assert json.loads((tmp_path / "logs" / "ppo" / "tiny_s0" / "summary.json").read_text())["algo"] == "ppo"
    policy = load_policy(model_dir / "model_final.pt")
    assert (policy.name, policy.mode, summary["eval_mode"]) == ("ppo", "sample", "sample")

    rows, train_rows = _rows(tmp_path)
    assert list(rows[0])[-len(PPO_METRIC_COLUMNS):] == PPO_METRIC_COLUMNS
    per_rollout = cfg.num_epochs * cfg.minibatches_per_epoch  # 2 epochs x 4 minibatches
    assert [int(r["updates"]) for r in train_rows] == [per_rollout, 2 * per_rollout, 3 * per_rollout]
    assert [int(r["frames"]) for r in train_rows] == [100, 200, 300]
    assert summary["updates"] == 3 * per_rollout
    for r in train_rows:
        assert float(r["entropy"]) > 0 and float(r["value_loss"]) >= 0
        assert 0.0 <= float(r["clip_fraction"]) <= 1.0 and r["kl_approx"] != ""
        assert float(r["lr"]) == pytest.approx(cfg.lr)  # constant without annealing


def test_lr_annealing_is_linear_per_rollout(tmp_path):
    cfg = _tiny(anneal_lr=True, lr=3e-3)
    train_ppo(cfg, tmp_path)
    _, train_rows = _rows(tmp_path)
    assert [float(r["lr"]) for r in train_rows] == pytest.approx([3e-3, 2e-3, 1e-3])


def test_first_minibatch_ratio_is_one():
    """Stored action_log_prob must equal the collecting policy's log-prob, so the first PPO ratio is exactly 1."""
    torch.manual_seed(0)
    env = make_env(seed=0)
    actor, _ = make_actor(env.action_spec, [16])
    critic = make_critic([16])
    collector = Collector(env, actor, frames_per_batch=64, total_frames=64, auto_register_policy_transforms=False)
    batch = next(iter(collector))
    collector.shutdown()
    with torch.no_grad():
        GAE(gamma=0.99, lmbda=0.95, value_network=critic)(batch)
    out = ClipPPOLoss(actor, critic, clip_epsilon=0.2, entropy_coeff=0.01)(batch)
    assert float(out["clip_fraction"]) == 0.0
    assert float(out["kl_approx"]) == pytest.approx(0.0, abs=1e-6)
    assert float(out["mean_ratio"]) == pytest.approx(1.0, abs=1e-5)


def test_config_validation():
    for bad in (dict(clip_epsilon=0.0), dict(clip_epsilon=1.0), dict(gae_lambda=1.5), dict(entropy_coef=-0.1),
                dict(value_coef=0.0), dict(loss_critic_type="huber"), dict(num_epochs=0), dict(minibatch_size=0),
                dict(frames_per_batch=64, minibatch_size=128), dict(max_grad_norm=0.0), dict(algo="a2c")):
        with pytest.raises(ValueError):
            PPOConfig(**bad)
    assert PPOConfig(frames_per_batch=100, minibatch_size=30).minibatches_per_epoch == 3


def test_cli_dispatch(tmp_path, capsys):
    assert main(["ppo", "--name", "tiny", "--total-frames", "100", "--frames-per-batch", "50",
                 "--minibatch-size", "25", "--num-epochs", "1", "--hidden-sizes", "8", "--eval-episodes", "1",
                 "--anneal-lr", "true", "--root", str(tmp_path)]) == 0
    assert "tiny_s0" in capsys.readouterr().out
    cfg = PPOConfig.load(tmp_path / "logs" / "ppo" / "tiny_s0" / "config.json")
    assert (cfg.anneal_lr, cfg.minibatch_size, cfg.num_epochs) == (True, 25, 1)
