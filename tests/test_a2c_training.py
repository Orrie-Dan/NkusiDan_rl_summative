"""Card 12: A2C trainer (TorchRL A2CLoss + GAE).

Done-when ("a short run learns better than random") is checked by a manual run, too slow for the suite.
These tests cover the pieces that run relies on.
"""

import csv
import json

import pytest
import torch

from training.common import load_policy
from training.pg_training import A2C_METRIC_COLUMNS, A2CConfig, explained_variance, main, train_a2c


def _tiny(**kw) -> A2CConfig:
    base = dict(name="tiny", total_frames=300, frames_per_batch=100, hidden_sizes=[16],
                eval_every_frames=150, eval_episodes=1)
    return A2CConfig(**{**base, **kw})


def test_explained_variance():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(t, t) == pytest.approx(1.0)
    assert explained_variance(torch.full_like(t, t.mean()), t) == pytest.approx(0.0)
    assert explained_variance(-t, t) < 0


@pytest.mark.parametrize("variant", [dict(), dict(loss_critic_type="l2", normalize_advantages=False, entropy_coef=0.0)])
def test_tiny_run_trains_logs_and_checkpoints(tmp_path, variant):
    cfg = _tiny(**variant)
    summary = train_a2c(cfg, tmp_path)

    model_dir = tmp_path / "models" / "pg" / "a2c_tiny_s0"
    log_dir = tmp_path / "logs" / "a2c" / "tiny_s0"
    assert {p.name for p in model_dir.iterdir()} == {"model.pt", "model_final.pt", "config.json"}
    assert A2CConfig.load(model_dir / "config.json") == cfg
    assert json.loads((log_dir / "summary.json").read_text())["algo"] == "a2c"
    policy = load_policy(model_dir / "model_final.pt")
    assert (policy.name, policy.mode, summary["eval_mode"]) == ("a2c", "sample", "sample")

    rows = list(csv.DictReader(open(log_dir / "train_log.csv", encoding="utf-8")))
    assert list(rows[0])[-len(A2C_METRIC_COLUMNS):] == A2C_METRIC_COLUMNS
    train_rows = [r for r in rows if r["updates"] != ""]
    # one gradient step per collector batch of exactly frames_per_batch decisions
    assert [int(r["updates"]) for r in train_rows] == [1, 2, 3]
    assert [int(r["frames"]) for r in train_rows] == [100, 200, 300]
    assert summary["frames"] == 300 and summary["updates"] == 3
    for r in train_rows:
        assert float(r["entropy"]) > 0 and float(r["value_loss"]) >= 0 and float(r["grad_norm"]) >= 0
        assert r["policy_loss"] != "" and r["explained_variance"] != ""


def test_update_changes_both_actor_and_critic(tmp_path, monkeypatch):
    """The optimizer must reach the networks the collector and the checkpoint use (shared parameters)."""
    import training.pg_training as pg

    built = {}
    real_actor, real_critic = pg.make_actor, pg.make_critic

    def spy_actor(*a, **k):
        actor, net = real_actor(*a, **k)
        built["actor"] = (net, [p.detach().clone() for p in net.parameters()])
        return actor, net

    def spy_critic(*a, **k):
        critic = real_critic(*a, **k)
        built["critic"] = (critic, [p.detach().clone() for p in critic.parameters()])
        return critic

    monkeypatch.setattr(pg, "make_actor", spy_actor)
    monkeypatch.setattr(pg, "make_critic", spy_critic)
    train_a2c(_tiny(total_frames=100, eval_every_frames=10_000), tmp_path)
    for name, (module, before) in built.items():
        assert any(not torch.equal(b, p) for b, p in zip(before, module.parameters())), f"{name} never updated"

    saved = load_policy(tmp_path / "models" / "pg" / "a2c_tiny_s0" / "model_final.pt").net
    assert all(torch.equal(a, b) for a, b in zip(saved.state_dict().values(),
                                                  built["actor"][0].state_dict().values()))


def test_config_validation():
    for bad in (dict(gae_lambda=1.5), dict(gae_lambda=-0.1), dict(entropy_coef=-0.1), dict(value_coef=0.0),
                dict(loss_critic_type="huber"), dict(frames_per_batch=1), dict(max_grad_norm=0.0),
                dict(algo="ppo")):
        with pytest.raises(ValueError):
            A2CConfig(**bad)


def test_cli_dispatch(tmp_path, capsys):
    assert main(["a2c", "--name", "tiny", "--total-frames", "100", "--frames-per-batch", "50",
                 "--hidden-sizes", "8", "--eval-episodes", "1", "--gae-lambda", "0.9",
                 "--root", str(tmp_path)]) == 0
    assert "tiny_s0" in capsys.readouterr().out
    cfg = A2CConfig.load(tmp_path / "logs" / "a2c" / "tiny_s0" / "config.json")
    assert (cfg.gae_lambda, cfg.hidden_sizes, cfg.frames_per_batch) == (0.9, [8], 50)
