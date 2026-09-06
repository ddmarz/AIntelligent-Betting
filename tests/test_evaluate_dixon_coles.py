import math
import numpy as np
import pandas as pd
from evaluate_dixon_coles import (
    TARGET_LEAGUES,
    XI_CANDIDATES,
    load_target_matches,
    implied_probabilities,
    log_loss,
    walk_forward_evaluate,
    run_xi_grid_search,
)


def test_implied_probabilities_removes_bookmaker_margin():
    p_home, p_draw, p_away = implied_probabilities(2.0, 3.0, 4.0)
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-9
    assert p_home > p_draw > p_away


def test_load_target_matches_filters_to_target_leagues(tmp_path):
    csv_path = tmp_path / "matches_clean.csv"
    pd.DataFrame({
        "Division": ["I1", "N1", "E0"],
        "MatchDate": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Season": ["2019-2020", "2019-2020", "2019-2020"],
        "HomeTeam": ["A", "X", "C"],
        "AwayTeam": ["B", "Y", "D"],
        "FTHome": [1, 2, 0],
        "FTAway": [0, 2, 0],
        "FTResult": ["H", "D", "D"],
        "OddHome": [2.0, 2.0, 2.0],
        "OddDraw": [3.0, 3.0, 3.0],
        "OddAway": [4.0, 4.0, 4.0],
        "HomeShots": [10, 12, 8],
        "AwayShots": [8, 9, 11],
        "HomeTarget": [4, 5, 3],
        "AwayTarget": [3, 4, 5],
    }).to_csv(csv_path, index=False)

    matches = load_target_matches(str(csv_path))

    assert set(matches["Division"]) == {"I1", "E0"}
    assert set(TARGET_LEAGUES) >= set(matches["Division"])
    assert pd.api.types.is_datetime64_any_dtype(matches["MatchDate"])


def test_log_loss_picks_probability_of_actual_outcome():
    assert log_loss(0.5, 0.3, 0.2, "H") == -math.log(0.5)
    assert log_loss(0.5, 0.3, 0.2, "D") == -math.log(0.3)
    assert log_loss(0.5, 0.3, 0.2, "A") == -math.log(0.2)


def test_log_loss_confident_correct_prediction_scores_near_zero():
    assert log_loss(0.99, 0.005, 0.005, "H") < 0.02


def _fake_league(n_seasons=5, teams=("A", "B", "C", "D")):
    np.random.seed(1)
    rows = []
    for season_idx in range(n_seasons):
        season_start = pd.Timestamp("2015-08-01") + pd.DateOffset(years=season_idx)
        season_label = f"{season_start.year}-{season_start.year + 1}"
        for round_num in range(10):
            date = season_start + pd.Timedelta(days=round_num * 7)
            shuffled = list(teams)
            np.random.shuffle(shuffled)
            for i in range(0, len(shuffled), 2):
                h, a = shuffled[i], shuffled[i + 1]
                gh, ga = np.random.poisson(1.4), np.random.poisson(1.1)
                result = "H" if gh > ga else ("A" if ga > gh else "D")
                rows.append({
                    "Division": "I1", "MatchDate": date, "Season": season_label,
                    "HomeTeam": h, "AwayTeam": a, "FTHome": gh, "FTAway": ga,
                    "FTResult": result, "OddHome": 2.0, "OddDraw": 3.2, "OddAway": 3.8,
                })
    return pd.DataFrame(rows).sort_values("MatchDate").reset_index(drop=True)


def test_walk_forward_evaluate_skips_burn_in_seasons():
    league_matches = _fake_league(n_seasons=5)

    result = walk_forward_evaluate(league_matches, xi=0.001)

    evaluated_seasons = set(
        league_matches.set_index("MatchDate").loc[result["MatchDate"], "Season"]
    )
    burn_in_seasons = set(league_matches["Season"].unique()[:3])
    assert evaluated_seasons.isdisjoint(burn_in_seasons)
    assert not result.empty


def test_walk_forward_evaluate_returns_finite_log_losses():
    league_matches = _fake_league(n_seasons=5)

    result = walk_forward_evaluate(league_matches, xi=0.001)

    assert np.isfinite(result["model_log_loss"]).all()
    assert np.isfinite(result["market_log_loss"]).all()


def test_run_xi_grid_search_covers_all_candidates():
    matches = pd.concat([
        _fake_league(n_seasons=5).assign(Division="I1"),
        _fake_league(n_seasons=5).assign(Division="E0"),
    ], ignore_index=True)

    report = run_xi_grid_search(matches)

    assert set(report["xi"]) == set(XI_CANDIDATES)
    assert np.isfinite(report["model_log_loss"]).all()
