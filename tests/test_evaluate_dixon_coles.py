import pandas as pd
from evaluate_dixon_coles import TARGET_LEAGUES, load_target_matches, implied_probabilities


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
    }).to_csv(csv_path, index=False)

    matches = load_target_matches(str(csv_path))

    assert set(matches["Division"]) == {"I1", "E0"}
    assert set(TARGET_LEAGUES) >= set(matches["Division"])
    assert pd.api.types.is_datetime64_any_dtype(matches["MatchDate"])
