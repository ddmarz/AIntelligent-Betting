import numpy as np
import pandas as pd

from dataset.dataset import (
    PREDICTED_STATS_COLUMNS,
    STATS_COLUMNS,
    add_availability_flags,
    drop_missing_stats,
)


def _stats_matches():
    labels = ["none", "ht_only", "full", "partial", "no_total_shots"]
    data = {
        "Division": ["I1"] * len(labels),
        "HomeTeam": labels,
        "AwayTeam": ["Opponent"] * len(labels),
    }
    for col in STATS_COLUMNS:
        data[col] = [np.nan] * len(labels)
    matches = pd.DataFrame(data)
    matches["HTResult"] = matches["HTResult"].astype(object)

    rows = {label: i for i, label in enumerate(labels)}

    matches.loc[rows["ht_only"], ["HTHome", "HTAway", "HTResult"]] = [1, 0, "H"]

    for col in STATS_COLUMNS:
        matches.loc[rows["full"], col] = "D" if col == "HTResult" else 3
    matches.loc[rows["full"], "HomeTarget"] = 0

    matches.loc[rows["partial"], ["HomeTarget", "AwayTarget", "HomeCorners"]] = [5, 4, 6]

    matches.loc[rows["no_total_shots"], ["HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners"]] = [5, 4, 6, 3]

    return matches


def test_predicted_stats_columns():
    assert PREDICTED_STATS_COLUMNS == ["HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners"]


def test_drop_missing_stats_removes_only_rows_without_any_stat():
    result = drop_missing_stats(_stats_matches())
    assert list(result["HomeTeam"]) == ["ht_only", "full", "partial", "no_total_shots"]


def test_drop_missing_stats_uses_only_present_columns():
    matches = _stats_matches().drop(columns=["HTHome", "HTAway", "HTResult"])
    result = drop_missing_stats(matches)
    assert list(result["HomeTeam"]) == ["full", "partial", "no_total_shots"]


def test_has_stats_requires_all_predicted_columns():
    result = add_availability_flags(drop_missing_stats(_stats_matches())).set_index("HomeTeam")
    assert result.loc["ht_only", "has_stats"] == False
    assert result.loc["full", "has_stats"] == True  # HomeTarget=0 è un valore valido
    assert result.loc["partial", "has_stats"] == False
    assert result.loc["no_total_shots", "has_stats"] == True


def test_missing_stats_are_still_filled():
    result = add_availability_flags(drop_missing_stats(_stats_matches())).set_index("HomeTeam")
    assert result.loc["ht_only", "HomeTarget"] == 0
    assert result.loc["partial", "AwayCorners"] == 0
    assert result[STATS_COLUMNS].notna().all().all()
