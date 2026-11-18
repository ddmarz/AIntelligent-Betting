import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
from scipy.stats import nbinom, poisson

from dl_model.calibration import (
    build_events,
    calibration_report,
    confidence_summary,
    expected_calibration_error,
    find_run,
    load_run,
    market_prob_over,
    nb_pmf,
    prob_over_goals,
    prob_over_single,
    prob_over_total,
    reliability_table,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_frames():
    """Per-match CSV frames of a run with 3 matches (columns written out: importing
    dl_model.evaluation would pull in torch). The third match has missing O/U odds (0.0)."""
    ids = {
        "Division": ["I1", "E0", "SP1"],
        "Season": ["2019-2020"] * 3,
        "MatchDate": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "HomeTeam": ["Juventus", "Arsenal", "Getafe"],
        "AwayTeam": ["Milan", "Chelsea", "Betis"],
    }
    goals = pd.DataFrame({
        **ids,
        "FTResult": ["H", "D", "A"], "FTHome": [2, 1, 0], "FTAway": [1, 1, 3],
        "OddHome": [1.8, 2.2, 2.6], "OddDraw": [3.5, 3.1, 3.2], "OddAway": [4.0, 3.4, 2.8],
        "Over25": [1.9, 2.1, 0.0], "Under25": [1.95, 1.75, 0.0],
        "lambda_home": [1.6, 1.2, 1.1], "lambda_away": [1.0, 1.1, 1.4],
        "model_p_home": [0.5, 0.38, 0.33], "model_p_draw": [0.27, 0.3, 0.28], "model_p_away": [0.23, 0.32, 0.39],
        "market_p_home": [0.53, 0.43, 0.37], "market_p_draw": [0.27, 0.3, 0.29], "market_p_away": [0.2, 0.27, 0.34],
        "model_log_loss": [0.69, 1.2, 0.94], "market_log_loss": [0.63, 1.2, 1.08],
    })
    counts = pd.DataFrame({
        **ids,
        "HomeTarget": [6, 3, 2], "AwayTarget": [4, 4, 5], "HomeCorners": [7, 4, 3], "AwayCorners": [3, 5, 6],
        "pred_home_shots_on_target": [5.2, 4.4, 3.9], "pred_away_shots_on_target": [3.8, 4.1, 4.6],
        "pred_home_corners": [5.9, 5.1, 4.8], "pred_away_corners": [4.2, 4.6, 5.0],
        "disp_home_shots_on_target": [8.0, 7.5, 7.0], "disp_away_shots_on_target": [7.0, 7.5, 8.0],
        "disp_home_corners": [6.0, 5.5, 5.0], "disp_away_corners": [5.0, 5.5, 6.0],
        "shots_on_target_nll": [4.4, 4.3, 4.6], "corners_nll": [4.7, 4.6, 4.9],
    })
    return goals, counts


def test_prob_over_single_matches_scipy_nbinom():
    p = prob_over_single(np.array([5.0]), np.array([3.0]), 4.5)
    assert p[0] == pytest.approx(nbinom.sf(4, 3.0, 3.0 / 8.0))


def test_prob_over_total_tends_to_poisson_sum_with_huge_dispersion():
    p = prob_over_total(np.array([5.0]), np.array([1e6]), np.array([4.0]), np.array([1e6]), 9.5)
    assert p[0] == pytest.approx(poisson.sf(9, 9.0), abs=1e-3)


def test_prob_over_total_decreases_with_line():
    args = (np.array([5.5]), np.array([6.0]), np.array([4.5]), np.array([6.0]))
    probs = [prob_over_total(*args, line)[0] for line in (6.5, 8.5, 10.5, 12.5)]
    assert probs == sorted(probs, reverse=True)


def test_nb_pmf_support_holds_almost_all_mass():
    assert nb_pmf(np.array([15.0]), np.array([5.0])).sum() >= 0.9999


def test_prob_over_goals_is_poisson_of_summed_lambdas():
    assert prob_over_goals(np.array([1.5]), np.array([1.0]), 2.5)[0] == pytest.approx(poisson.sf(2, 2.5))


def test_market_prob_over_removes_margin():
    assert market_prob_over(np.array([1.9]), np.array([1.9]))[0] == pytest.approx(0.5)


def test_wilson_interval_known_values_and_empty():
    assert wilson_interval(3, 4) == pytest.approx((0.3006, 0.9544), abs=1e-4)
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def test_reliability_table_perfect_calibration_and_edges():
    pred = np.array([0.75, 0.75, 0.75, 0.75, 1.0])
    outcome = np.array([1, 1, 1, 0, 1])
    table = reliability_table(pred, outcome)
    assert len(table) == 10
    row = table[np.isclose(table["bin_low"], 0.7)].iloc[0]
    assert (row["n"], row["mean_pred"], row["obs_freq"]) == (4, pytest.approx(0.75), pytest.approx(0.75))
    assert table.iloc[-1]["n"] == 1  # pred = 1.0 nell'ultima fascia
    assert table.iloc[0]["n"] == 0 and np.isnan(table.iloc[0]["obs_freq"])
    assert expected_calibration_error(table) == pytest.approx(0.0)


def test_reliability_table_small_bin_has_wide_interval():
    table = reliability_table(np.array([0.55, 0.55, 0.55]), np.array([1, 0, 1]))
    row = table[table["n"] == 3].iloc[0]
    assert row["ci_high"] - row["ci_low"] > 0.5


def test_confidence_summary_reports_both_sides():
    s = confidence_summary(np.array([0.8, 0.8, 0.2, 0.5]), np.array([1, 0, 0, 1]))
    assert s["share_confident"] == pytest.approx(0.75)
    assert (s["over_n"], s["over_mean_pred"], s["over_obs"]) == (2, pytest.approx(0.8), pytest.approx(0.5))
    assert (s["under_n"], s["under_mean_pred"], s["under_obs"]) == (1, pytest.approx(0.8), pytest.approx(1.0))


def test_calibration_module_does_not_import_torch():
    code = "import sys, dl_model.calibration; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)


def test_build_events_families_and_counts():
    events = build_events(*_run_frames())
    model = events[events["source"] == "model"]
    assert set(model["family"]) == {"corners_total", "corners_team", "sot_total", "sot_team", "goals_ou", "1x2"}
    assert len(model[model["family"] == "corners_total"]) == 3 * 7
    assert len(model[model["family"] == "corners_team"]) == 3 * 2 * 5
    assert len(model[model["family"] == "1x2"]) == 3 * 3


def test_build_events_corner_total_matches_probability_function():
    goals, counts = _run_frames()
    events = build_events(goals, counts)
    row = events[(events["family"] == "corners_total") & (events["line"] == 9.5)].iloc[0]
    expected = prob_over_total(counts["pred_home_corners"].values[:1], counts["disp_home_corners"].values[:1],
                               counts["pred_away_corners"].values[:1], counts["disp_away_corners"].values[:1], 9.5)[0]
    assert row["pred"] == pytest.approx(expected)
    assert row["outcome"] == int(counts["HomeCorners"].iloc[0] + counts["AwayCorners"].iloc[0] > 9.5)


def test_build_events_market_skips_missing_ou_odds():
    events = build_events(*_run_frames())
    market_ou = events[(events["source"] == "market") & (events["family"] == "goals_ou")]
    assert len(market_ou) == 2  # la terza partita ha quote 0
    assert set(market_ou["line"]) == {2.5}
    assert np.isfinite(market_ou["pred"]).all()


def test_calibration_report_one_summary_row_per_group():
    summary, bins = calibration_report(build_events(*_run_frames()))
    assert len(summary) == len(summary[["family", "market", "line", "source"]].drop_duplicates())
    assert {"n", "ece", "share_confident", "over_obs", "under_obs"} <= set(summary.columns)
    assert len(bins) == 10 * len(summary)


def test_load_run_rejects_old_run_without_new_columns(tmp_path):
    goals, counts = _run_frames()
    goals.drop(columns=["Over25"]).to_csv(tmp_path / "multitask_old_goals.csv", index=False)
    counts.to_csv(tmp_path / "multitask_old_counts.csv", index=False)
    with pytest.raises(ValueError, match="rilancia il training"):
        load_run(tmp_path, "multitask_old")


def test_find_run_picks_latest(tmp_path):
    for stamp in ("20261117-100000-000000", "20261118-090000-000000"):
        (tmp_path / f"multitask_{stamp}_goals.csv").write_text("x")
    assert find_run(tmp_path) == "multitask_20261118-090000-000000"


def test_main_writes_summary_bins_and_plots(tmp_path):
    goals, counts = _run_frames()
    goals.to_csv(tmp_path / "multitask_20261117-100000-000000_goals.csv", index=False)
    counts.to_csv(tmp_path / "multitask_20261117-100000-000000_counts.csv", index=False)
    # Run the command in its own process, as it is used for real: in the test session other test
    # modules have loaded torch, and torch + numpy/MKL in one process abort on the first matplotlib draw
    # (OpenMP error #15 in this environment).
    subprocess.run(
        [sys.executable, "-m", "dl_model.calibration", "--checkpoint-dir", str(tmp_path)],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )
    out = tmp_path / "multitask_20261117-100000-000000_calibration"
    assert (out / "summary.csv").exists() and (out / "bins.csv").exists()
    assert len(list(out.glob("reliability_*.png"))) == 6


def test_reliability_table_keeps_tiny_negative_probability_in_first_bin():
    table = reliability_table(np.array([-1e-16, 0.05]), np.array([0, 0]))
    assert table.iloc[0]["n"] == 2


def test_calibration_report_shows_base_rate_and_brier_skill():
    rng = np.random.default_rng(0)
    outcome = (rng.random(1000) < 0.86).astype(int)
    events = pd.DataFrame({
        "family": "corners_total", "market": ["base"] * 1000 + ["perfect"] * 1000, "line": 6.5, "source": "model",
        "pred": np.concatenate([np.full(1000, outcome.mean()), outcome.astype(float)]),
        "outcome": np.concatenate([outcome, outcome]),
    })
    summary, _ = calibration_report(events)
    base = summary[summary["market"] == "base"].iloc[0]
    perfect = summary[summary["market"] == "perfect"].iloc[0]
    assert base["base_rate"] == pytest.approx(outcome.mean())
    assert base["brier_skill"] == pytest.approx(0.0, abs=1e-9)   # always predicting the average adds nothing
    assert perfect["brier_skill"] == pytest.approx(1.0)
