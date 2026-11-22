"""Card 15: sweep configs and the parallel, resumable runner.

The done-when check (a dry run of all 40 entries) is a manual run, too slow for the suite. These tests cover the
config table and the runner's skip / stale / failure-isolation logic with tiny real runs.
"""

import csv
import json
from dataclasses import asdict

import pytest

import main as sweep
from training import ALGOS, log_dir_for
from main import (
    ALL_ENTRIES, CONFIG_CLASSES, SWEEP, build_config, dry_run_config, job_state, run_jobs, select_entries,
)


def test_ten_configs_per_algo_with_unique_names():
    assert set(SWEEP) == set(ALGOS)
    for algo, entries in SWEEP.items():
        assert [e.index for e in entries] == list(range(10))
        assert all(e.algo == algo for e in entries)
    assert len(ALL_ENTRIES) == 40


@pytest.mark.parametrize("name", sorted(ALL_ENTRIES))
def test_each_config_is_valid_and_changes_one_factor_from_base(name):
    entry = ALL_ENTRIES[name]
    cfg = build_config(entry)
    base = asdict(CONFIG_CLASSES[entry.algo](name=entry.name))
    diff = {k for k, v in asdict(cfg).items() if base[k] != v}
    assert diff == set(entry.overrides)  # every override really changes the base value
    assert entry.rationale and entry.factor
    if entry.index == 0:
        assert entry.overrides == {} and entry.factor == "base"
    else:
        assert len(entry.overrides) == 1  # one factor at a time
    assert cfg.total_frames == 200_000 and cfg.name == name and cfg.seed == 0


def test_no_duplicate_configs_within_an_algo():
    for entries in SWEEP.values():
        changes = [json.dumps(e.overrides, sort_keys=True) for e in entries]
        assert len(set(changes)) == len(changes)


def test_dry_run_configs_are_tiny_and_dqn_learns_within_budget():
    for entry in ALL_ENTRIES.values():
        cfg = dry_run_config(entry)
        assert cfg.total_frames == sweep.DRY_RUN_FRAMES
        if entry.algo == "dqn":
            assert cfg.learning_starts < cfg.total_frames


def test_select_entries_filters():
    assert [e.name for e in select_entries(["ppo"], ["c00", "3"])] == ["ppo_c00", "ppo_c03"]
    assert len(select_entries()) == 40
    with pytest.raises(ValueError):
        select_entries(["sac"])


def _tiny(name="ppo_c00", seed=0, **kw):
    base = dict(total_frames=200, frames_per_batch=100, minibatch_size=50, num_epochs=1, hidden_sizes=[8],
                eval_every_frames=200, eval_episodes=1)
    return CONFIG_CLASSES["ppo"](name=name, seed=seed, **{**base, **kw})


def test_runner_runs_skips_and_reports_stale(tmp_path):
    cfgs = [_tiny(seed=0), _tiny(seed=1)]
    statuses = run_jobs(cfgs, tmp_path, workers=2, tag="t", echo=lambda *_: None)
    assert [s["status"] for s in statuses] == ["done", "done"]
    assert all(job_state(c, tmp_path) == "done" for c in cfgs)

    rows = list(csv.DictReader(open(tmp_path / "logs" / "sweep" / "t_results.csv", encoding="utf-8")))
    assert [(r["name"], r["seed"], r["factor"]) for r in rows] == [("ppo_c00", "0", "base"), ("ppo_c00", "1", "base")]
    assert json.loads((tmp_path / "logs" / "sweep" / "t_status.json").read_text())[0]["status"] == "done"

    # resume: nothing re-runs
    again = run_jobs(cfgs, tmp_path, workers=2, tag="t", echo=lambda *_: None)
    assert [s["status"] for s in again] == ["skipped", "skipped"]

    # same run id, different config: left alone unless overwrite
    changed = _tiny(seed=0, lr=1e-3)
    assert job_state(changed, tmp_path) == "stale"
    assert run_jobs([changed], tmp_path, tag="t2", echo=lambda *_: None)[0]["status"] == "stale"
    assert run_jobs([changed], tmp_path, tag="t2", overwrite=True, echo=lambda *_: None)[0]["status"] == "done"
    assert job_state(changed, tmp_path) == "done"


def test_one_failing_run_does_not_stop_the_others(tmp_path, monkeypatch):
    good, bad = _tiny(seed=0), _tiny(seed=1)
    real_command = sweep._command

    def command(cfg, config_path, root):
        if cfg.seed == 1:
            return [sweep.sys.executable, "-c", "raise SystemExit('boom')"]
        return real_command(cfg, config_path, root)

    monkeypatch.setattr(sweep, "_command", command)
    statuses = run_jobs([good, bad], tmp_path, workers=2, tag="t", echo=lambda *_: None)
    assert [s["status"] for s in statuses] == ["done", "failed"]
    assert "boom" in open(statuses[1]["log"], encoding="utf-8").read()
    assert not (log_dir_for("ppo", bad.run_id, tmp_path) / "summary.json").exists()


def test_cli_list_prints_all_configs(capsys):
    assert sweep.sweep_main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in ALL_ENTRIES)


def test_child_paths_are_relative_inside_the_project(tmp_path):
    inside = sweep.PROJECT_ROOT / "logs" / "_scratch" / "x"
    assert sweep._child_path(inside).as_posix() == "logs/_scratch/x"
    assert sweep._child_path(tmp_path).is_absolute()  # outside the project: left absolute


def test_select_by_full_name_and_suffix_runs_get_own_names():
    picks = select_entries(configs=["dqn_c08", "reinforce_c03", "a2c_c03", "ppo_c04"])
    assert [e.name for e in picks] == ["dqn_c08", "reinforce_c03", "a2c_c03", "ppo_c04"]
    with pytest.raises(ValueError):
        select_entries(configs=["dqn_c99"])
    cfg = build_config(ALL_ENTRIES["ppo_c04"], seed=3, total_frames=600_000, name_suffix="_p2")
    assert (cfg.name, cfg.run_id, cfg.num_epochs, cfg.total_frames) == ("ppo_c04_p2", "ppo_c04_p2_s3", 10, 600_000)
    assert sweep.entry_for("ppo_c04_p2") is ALL_ENTRIES["ppo_c04"]
    assert sweep.entry_for("reinforce_c03") is ALL_ENTRIES["reinforce_c03"] and sweep.entry_for("tiny") is None
