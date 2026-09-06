import pandas as pd
import torch
from dl_model.data import (
    CONTINUOUS_COLUMNS,
    NUM_CONTINUOUS_FEATURES,
    UNKNOWN_TOKEN,
    build_vocab,
    encode_categorical,
    compute_normalization_stats,
    MatchDataset,
    load_all_matches,
)


def _sample_matches():
    data = {
        "Division": ["I1", "I1", "E0"],
        "MatchDate": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-02"]),
        "HomeTeam": ["Juventus", "Milan", "Arsenal"],
        "AwayTeam": ["Milan", "Roma", "Chelsea"],
        "FTHome": [2, 1, 0],
        "FTAway": [1, 1, 0],
        "FTResult": ["H", "D", "D"],
        "OddHome": [1.8, 2.5, 2.2],
        "OddDraw": [3.5, 3.2, 3.1],
        "OddAway": [4.0, 2.8, 3.4],
        "HomeShots": [15, 10, 8],
        "AwayShots": [9, 11, 7],
        "HomeCorners": [6, 4, 3],
        "AwayCorners": [3, 5, 2],
        "has_stats": [True, True, False],
    }
    for i, col in enumerate(CONTINUOUS_COLUMNS):
        data[col] = [0.1 + i * 0.01, -0.2 + i * 0.01, 0.0 + i * 0.01]
    return pd.DataFrame(data)


def test_build_vocab_assigns_unique_indices_with_unknown_at_zero():
    vocab = build_vocab(["Juventus", "Milan", "Juventus"])
    assert vocab[UNKNOWN_TOKEN] == 0
    assert len(vocab) == 3
    assert len(set(vocab.values())) == 3


def test_encode_categorical_falls_back_to_unknown():
    vocab = build_vocab(["Juventus", "Milan"])
    assert encode_categorical("NeverSeenTeam", vocab) == vocab[UNKNOWN_TOKEN]
    assert encode_categorical("Juventus", vocab) == vocab["Juventus"]


def test_compute_normalization_stats_returns_mean_and_std_per_column():
    matches = _sample_matches()
    means, stds = compute_normalization_stats(matches)
    assert set(means.index) == set(CONTINUOUS_COLUMNS)
    assert (stds > 0).all()


def test_match_dataset_returns_expected_tensor_shapes_and_values():
    matches = _sample_matches()
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(matches)

    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    assert len(dataset) == 3

    continuous, division_idx, home_idx, away_idx, targets = dataset[0]

    assert continuous.shape == (NUM_CONTINUOUS_FEATURES,)
    assert division_idx.dtype == torch.long
    assert home_idx.item() == team_vocab["Juventus"]
    assert away_idx.item() == team_vocab["Milan"]
    assert targets["goals_home"].item() == 2.0
    assert targets["goals_away"].item() == 1.0
    assert targets["has_stats"].item() is True


def test_load_all_matches_keeps_all_leagues(tmp_path):
    csv_path = tmp_path / "matches_clean.csv"
    _sample_matches().to_csv(csv_path, index=False)

    matches = load_all_matches(str(csv_path))

    assert set(matches["Division"]) == {"I1", "E0"}
    assert pd.api.types.is_datetime64_any_dtype(matches["MatchDate"])
