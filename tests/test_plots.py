"""Card 20: `main.py plots` rebuilds every figure and table from result files only."""

from pathlib import Path

import pandas as pd
import pytest

import main

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("fig1_learning_curves", "fig2_dqn_objective", "fig3_pg_entropy", "fig4_convergence",
           "fig5_generalization", "fig6_kpi_breakdown", "fig7_phase1_configs", "fig8_sensitivity")


def test_binned_averages_per_bin_and_drops_nans():
    log = pd.DataFrame({"frames": [0, 5_000, 10_000, 15_000, 20_000], "x": [1.0, 3.0, None, 4.0, 6.0]})
    s = main._binned(log, "x", step=10_000)
    assert s.to_dict() == {0: 2.0, 10_000: 4.0, 20_000: 6.0}


def test_seed_frame_keeps_only_bins_shared_by_every_run():
    a = pd.DataFrame({"frames": [0, 10_000, 20_000], "x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"frames": [0, 10_000], "x": [5.0, 6.0]})
    wide = main._seed_frame([a, b], "x")
    assert list(wide.index) == [0, 10_000]
    assert wide.mean(axis=1).tolist() == [3.0, 4.0]


@pytest.mark.skipif(not (ROOT / "logs" / "eval" / "final" / "base" / "episodes.csv").exists(),
                    reason="needs the committed Phase 1/2 logs and the card 19 evaluation")
def test_make_plots_writes_every_figure_table_and_caption(tmp_path):
    files = main.make_plots(ROOT, tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in files}
    for fig in FIGURES:
        assert f"{fig}.png" in names
    for algo in main.PLOT_ALGOS:
        assert f"tables/hyperparameters_{algo}.csv" in names
        assert f"tables/base_config_{algo}.md" in names
    assert "tables/final_eval_base.csv" in names and "tables/generalization.csv" in names

    captions = (tmp_path / "captions.md").read_text(encoding="utf-8")
    assert all(f"**{fig}.png**" in captions for fig in FIGURES)

    # every algorithm table has its 10 configs and exactly one picked config, matching Phase 2
    p2 = pd.read_csv(ROOT / "logs" / "sweep" / "phase2_results.csv")
    for algo in main.PLOT_ALGOS:
        t = pd.read_csv(tmp_path / "tables" / f"hyperparameters_{algo}.csv")
        assert len(t) == 10
        picked = t.loc[t["picked"] == "yes", "config"].tolist()
        assert picked == [p2.loc[p2.algo == algo, "name"].iloc[0].removesuffix("_p2")]

    # pooled RL rows cover 5 seeds x 50 episodes; baselines 50 episodes
    base = pd.read_csv(tmp_path / "tables" / "final_eval_base.csv").set_index("policy")
    assert base.loc["DQN (5 seeds)", "episodes"] == 250
    assert base.loc["rolling_horizon", "episodes"] == 50
