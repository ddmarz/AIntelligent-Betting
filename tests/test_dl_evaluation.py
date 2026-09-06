import math
import numpy as np
import pandas as pd
from dl_model.data import CONTINUOUS_COLUMNS, build_vocab, compute_normalization_stats
from dl_model.model import MultiTaskMatchModel
from dl_model.evaluation import (
    poisson_score_matrix,
    match_outcome_probs,
    implied_probabilities,
    log_loss,
    evaluate_goals,
    evaluate_counts,
)


def test_poisson_score_matrix_sums_to_one_and_has_no_tau_artifact():
    matrix = poisson_score_matrix(1.5, 1.2, max_goals=10)
    assert matrix.shape == (11, 11)
    assert abs(matrix.sum() - 1.0) < 1e-6
    assert matrix[0, 0] > 0


def test_match_outcome_probs_sum_to_one_and_favor_stronger_side():
    matrix = poisson_score_matrix(2.5, 0.8, max_goals=10)
    p_home, p_draw, p_away = match_outcome_probs(matrix)
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-6
    assert p_home > p_away


def test_implied_probabilities_removes_bookmaker_margin():
    p_home, p_draw, p_away = implied_probabilities(2.0, 3.0, 4.0)
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-9
    assert p_home > p_draw > p_away


def test_log_loss_picks_probability_of_actual_outcome():
    assert log_loss(0.5, 0.3, 0.2, "H") == -math.log(0.5)
    assert log_loss(0.5, 0.3, 0.2, "D") == -math.log(0.3)
    assert log_loss(0.5, 0.3, 0.2, "A") == -math.log(0.2)


def _sample_matches():
    data = {
        "Division": ["I1", "E0"],
        "MatchDate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "HomeTeam": ["Juventus", "Arsenal"],
        "AwayTeam": ["Milan", "Chelsea"],
        "FTHome": [2, 0],
        "FTAway": [1, 0],
        "FTResult": ["H", "D"],
        "OddHome": [1.8, 2.2],
        "OddDraw": [3.5, 3.1],
        "OddAway": [4.0, 3.4],
        "HomeShots": [15, 8],
        "AwayShots": [9, 7],
        "HomeCorners": [6, 3],
        "AwayCorners": [3, 2],
        "has_stats": [True, True],
    }
    for i, col in enumerate(CONTINUOUS_COLUMNS):
        data[col] = [0.1 + i * 0.01, -0.2 + i * 0.01]
    return pd.DataFrame(data)


def _sample_model_and_vocabs(matches):
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(matches)
    model = MultiTaskMatchModel(num_divisions=len(division_vocab), num_teams=len(team_vocab))
    return model, division_vocab, team_vocab, means, stds


def test_evaluate_goals_returns_expected_columns():
    matches = _sample_matches()
    model, division_vocab, team_vocab, means, stds = _sample_model_and_vocabs(matches)

    result = evaluate_goals(model, matches, division_vocab, team_vocab, means, stds)

    assert set(result.columns) == {"MatchDate", "model_log_loss", "market_log_loss"}
    assert len(result) == 2
    assert np.isfinite(result["model_log_loss"]).all()


def test_evaluate_counts_returns_expected_columns():
    matches = _sample_matches()
    model, division_vocab, team_vocab, means, stds = _sample_model_and_vocabs(matches)

    result = evaluate_counts(model, matches, division_vocab, team_vocab, means, stds)

    assert set(result.columns) == {"shots_nll", "corners_nll"}
    assert len(result) == 2
    assert np.isfinite(result["shots_nll"]).all()
    assert np.isfinite(result["corners_nll"]).all()
