"""Card 11: REINFORCE trainer (hand-written loss).

Done-when ("a short run learns better than random") is checked by a manual run, too slow for the suite.
These tests cover the pieces that run relies on.
"""

import csv
import json

import pytest
import torch
from tensordict import TensorDict

from environment.custom_env import OBS_DIM
from environment.simulation import N_ACTIONS
from training.common import load_policy, make_actor, make_env, mlp
from training.pg_training import (
    REINFORCE_METRIC_COLUMNS,
    ReinforceConfig,
    collect_episodes,
    main,
    reinforce_loss,
    reward_to_go,
)


def _tiny(**kw) -> ReinforceConfig:
    base = dict(name="tiny", total_frames=300, episodes_per_update=2, hidden_sizes=[16],
                eval_every_frames=150, eval_episodes=1)
    return ReinforceConfig(**{**base, **kw})


def test_reward_to_go_is_discounted_suffix_sum():
    r = torch.tensor([[1.0], [2.0], [3.0]])
    assert torch.allclose(reward_to_go(r, 0.5), torch.tensor([1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.0]))
    assert torch.allclose(reward_to_go(r, 1.0), torch.tensor([6.0, 5.0, 3.0]))


def test_collected_episodes_are_complete_valid_and_stochastic():
    torch.manual_seed(0)
    env = make_env(seed=0)
    actor, _ = make_actor(env.action_spec, [16])
    episodes = collect_episodes(env, actor, 3)
    for ep in episodes:
        assert bool(ep["next", "done"][-1]) and not ep["next", "done"][:-1].any()  # exactly one episode each
        taken = ep["action_mask"].gather(-1, ep["action"].unsqueeze(-1))
        assert taken.all(), "an invalid action was sampled"
    # sampling, not greedy: the actions differ from the argmax of the logits somewhere
    batch = torch.cat(episodes)
    greedy = batch["logits"].masked_fill(~batch["action_mask"], float("-inf")).argmax(-1)
    assert (greedy != batch["action"]).any()


def test_loss_gradient_raises_log_prob_of_positive_advantage_actions():
    torch.manual_seed(0)
    net = mlp(N_ACTIONS, [16])
    n = 8
    mask = torch.ones(n, N_ACTIONS, dtype=torch.bool)
    mask[:, 5:] = False  # only actions 0-4 valid
    batch = TensorDict({"observation": torch.rand(n, OBS_DIM), "action_mask": mask,
                        "action": torch.full((n,), 2)}, batch_size=[n])
    adv = torch.ones(n)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    def logp():
        return torch.distributions.Categorical(logits=net(batch["observation"]).masked_fill(~mask, -1e9)) \
            .log_prob(batch["action"]).mean().item()

    before = logp()
    loss, policy_loss, entropy = reinforce_loss(net, batch, adv, entropy_coef=0.0)
    assert torch.isfinite(loss) and torch.isfinite(entropy)
    assert entropy.item() <= torch.log(torch.tensor(5.0)).item() + 1e-5  # entropy only over the 5 valid actions
    opt.zero_grad(); loss.backward(); opt.step()
    assert logp() > before

    # negative advantage pushes the other way
    before = logp()
    loss, _, _ = reinforce_loss(net, batch, -adv, entropy_coef=0.0)
    opt.zero_grad(); loss.backward(); opt.step()
    assert logp() < before


@pytest.mark.parametrize("variant", [dict(), dict(baseline="moving_average"),
                                     dict(baseline="none", normalize_advantages=False, entropy_coef=0.0)])
def test_tiny_run_trains_logs_and_checkpoints(tmp_path, variant):
    cfg = _tiny(**variant)
    summary = train_and_check_files(tmp_path, cfg)
    rows = list(csv.DictReader(open(tmp_path / "logs" / "reinforce" / "tiny_s0" / "train_log.csv", encoding="utf-8")))
    assert list(rows[0])[-len(REINFORCE_METRIC_COLUMNS):] == REINFORCE_METRIC_COLUMNS
    train_rows = [r for r in rows if r["updates"] != ""]
    assert [int(r["updates"]) for r in train_rows] == list(range(1, len(train_rows) + 1))
    assert [int(r["episodes"]) for r in train_rows] == [2 * i for i in range(1, len(train_rows) + 1)]
    frames = [int(r["frames"]) for r in train_rows]
    assert frames[-1] >= cfg.total_frames > frames[-2]  # stop at the first batch that reaches the budget
    assert summary["frames"] == frames[-1] and summary["updates"] == len(train_rows)
    assert all(float(r["entropy"]) > 0 for r in train_rows)
    has_value_loss = cfg.baseline == "value"
    assert all((r["value_loss"] != "") == has_value_loss for r in train_rows)
    if cfg.baseline == "none":
        assert all(float(r["baseline_mean"]) == 0.0 for r in train_rows)


def train_and_check_files(tmp_path, cfg):
    from training.pg_training import train_reinforce

    summary = train_reinforce(cfg, tmp_path)
    model_dir = tmp_path / "models" / "pg" / "reinforce_tiny_s0"
    assert {p.name for p in model_dir.iterdir()} == {"model.pt", "model_final.pt", "config.json"}
    assert ReinforceConfig.load(model_dir / "config.json") == cfg
    assert json.loads((tmp_path / "logs" / "reinforce" / "tiny_s0" / "summary.json").read_text())["algo"] == "reinforce"
    policy = load_policy(model_dir / "model_final.pt")
    assert (policy.name, policy.mode, summary["eval_mode"]) == ("reinforce", "sample", "sample")
    return summary


def test_config_validation():
    for bad in (dict(baseline="critic"), dict(baseline_momentum=1.0), dict(episodes_per_update=0),
                dict(value_steps_per_update=0), dict(entropy_coef=-0.1), dict(algo="ppo")):
        with pytest.raises(ValueError):
            ReinforceConfig(**bad)


def test_cli_dispatch(tmp_path, capsys):
    assert main([]) == 2 and main(["a2c"]) == 2
    assert main(["reinforce", "--name", "tiny", "--total-frames", "100", "--episodes-per-update", "1",
                 "--hidden-sizes", "8", "--eval-episodes", "1", "--baseline", "moving_average",
                 "--root", str(tmp_path)]) == 0
    assert "tiny_s0" in capsys.readouterr().out
    cfg = ReinforceConfig.load(tmp_path / "logs" / "reinforce" / "tiny_s0" / "config.json")
    assert (cfg.baseline, cfg.hidden_sizes, cfg.episodes_per_update) == ("moving_average", [8], 1)
