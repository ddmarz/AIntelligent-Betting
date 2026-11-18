import numpy as np
import pandas as pd
import pytest
import torch
from dl_model.data import CONTINUOUS_COLUMNS, compute_normalization_stats
from dl_model.evaluation import evaluate_counts, evaluate_goals
from dl_model.model import MultiTaskMatchModel
from dl_model.train import (
    SEED,
    build_model_and_vocabs,
    make_dataloader,
    train_epochs,
    compute_total_loss,
    run_walk_forward,
)


def _fake_matches(n_seasons=6, teams=("A", "B", "C", "D"), divisions=("I1", "E0")):
    np.random.seed(7)
    rows = []
    for season_idx in range(n_seasons):
        season_start = pd.Timestamp("2015-08-01") + pd.DateOffset(years=season_idx)
        season_label = f"{season_start.year}-{season_start.year + 1}"
        for division in divisions:
            for round_num in range(8):
                date = season_start + pd.Timedelta(days=round_num * 7)
                shuffled = list(teams)
                np.random.shuffle(shuffled)
                for i in range(0, len(shuffled), 2):
                    h, a = shuffled[i], shuffled[i + 1]
                    gh, ga = int(np.random.poisson(1.4)), int(np.random.poisson(1.1))
                    sh, sa = int(np.random.poisson(12)), int(np.random.poisson(10))
                    ch, ca = int(np.random.poisson(5)), int(np.random.poisson(4))
                    result = "H" if gh > ga else ("A" if ga > gh else "D")
                    row = {
                        "Division": division, "MatchDate": date, "Season": season_label,
                        "HomeTeam": h, "AwayTeam": a, "FTHome": gh, "FTAway": ga, "FTResult": result,
                        "OddHome": 2.0, "OddDraw": 3.2, "OddAway": 3.8, "Over25": 1.9, "Under25": 1.9,
                        "HomeShots": sh, "AwayShots": sa, "HomeCorners": ch, "AwayCorners": ca,
                        "HomeTarget": int(np.random.poisson(5)), "AwayTarget": int(np.random.poisson(4)),
                        "has_stats": True,
                    }
                    for col in CONTINUOUS_COLUMNS:
                        row[col] = float(np.random.randn())
                    rows.append(row)
    return pd.DataFrame(rows).sort_values("MatchDate").reset_index(drop=True)


def test_train_epochs_returns_mean_loss_per_epoch():
    matches = _fake_matches(n_seasons=2)
    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    loader = make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=32)

    losses = train_epochs(model, loader, torch.optim.Adam(model.parameters(), lr=0.01), num_epochs=3)

    assert len(losses) == 3
    assert all(np.isfinite(loss) for loss in losses)


def test_train_epochs_logs_one_line_per_epoch(capsys):
    matches = _fake_matches(n_seasons=2)
    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    loader = make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=32)

    losses = train_epochs(model, loader, torch.optim.Adam(model.parameters(), lr=0.01), num_epochs=2, desc="Burn-in")

    out = capsys.readouterr().out
    assert f"Burn-in | epoch 1/2 | loss {losses[0]:.4f}" in out
    assert f"Burn-in | epoch 2/2 | loss {losses[1]:.4f}" in out


def test_run_walk_forward_logs_burn_in_and_each_season(tmp_path, capsys):
    matches = _fake_matches(n_seasons=5)
    run_walk_forward(matches, burn_in_seasons=3, initial_epochs=1, fine_tune_epochs=1, checkpoint_dir=tmp_path)

    out = capsys.readouterr().out
    assert "Burn-in | epoch 1/1 | loss" in out
    assert "Season 2018-2019 (1/2) | epoch 1/1 | loss" in out
    assert "Season 2019-2020 (2/2) | epoch 1/1 | loss" in out


def test_train_epochs_reduces_average_loss():
    matches = _fake_matches(n_seasons=2)
    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    loader = make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def average_loss():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for continuous, division_idx, home_idx, away_idx, targets in loader:
                outputs = model(continuous, division_idx, home_idx, away_idx)
                total += compute_total_loss(outputs, targets).item()
                count += 1
        return total / count

    loss_before = average_loss()
    train_epochs(model, loader, optimizer, num_epochs=10)
    loss_after = average_loss()

    assert loss_after < loss_before


def test_run_walk_forward_returns_expected_report_shapes(tmp_path):
    matches = _fake_matches(n_seasons=6)

    goal_results, count_results = run_walk_forward(
        matches, burn_in_seasons=3, initial_epochs=2, fine_tune_epochs=1, checkpoint_dir=tmp_path
    )

    assert {"Season", "model_log_loss", "market_log_loss"} <= set(goal_results.columns)
    assert {"Season", "shots_on_target_nll", "corners_nll"} <= set(count_results.columns)
    assert not goal_results.empty
    assert not count_results.empty
    assert np.isfinite(goal_results["model_log_loss"]).all()


def test_seed_is_1926():
    assert SEED == 1926


def test_build_model_and_vocabs_normalizes_only_on_given_matches():
    matches = _fake_matches(n_seasons=6)
    burn_in = matches[matches["Season"].isin(matches["Season"].unique()[:3])]

    _model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches, normalization_matches=burn_in)

    expected_means, expected_stds = compute_normalization_stats(burn_in)
    pd.testing.assert_series_equal(means, expected_means)
    pd.testing.assert_series_equal(stds, expected_stds)
    assert set(team_vocab) >= set(matches["HomeTeam"])  # vocab still built on every match


def test_run_walk_forward_saves_checkpoint_with_burn_in_normalization(tmp_path):
    matches = _fake_matches(n_seasons=6)
    run_walk_forward(matches, burn_in_seasons=3, initial_epochs=1, fine_tune_epochs=1, checkpoint_dir=tmp_path)

    saved = list(tmp_path.glob("multitask_*.pt"))
    assert len(saved) == 1
    checkpoint = torch.load(saved[0], weights_only=True)

    burn_in = matches[matches["Season"].isin(matches["Season"].unique()[:3])]
    expected_means, expected_stds = compute_normalization_stats(burn_in)
    assert checkpoint["means"] == pytest.approx(expected_means.to_dict())
    assert checkpoint["stds"] == pytest.approx(expected_stds.to_dict())
    assert checkpoint["seed"] == 1926

    model = MultiTaskMatchModel(
        num_divisions=len(checkpoint["division_vocab"]), num_teams=len(checkpoint["team_vocab"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])


def test_run_walk_forward_is_reproducible(tmp_path):
    matches = _fake_matches(n_seasons=5)
    kwargs = dict(burn_in_seasons=3, initial_epochs=2, fine_tune_epochs=1, checkpoint_dir=tmp_path)

    first_goals, first_counts = run_walk_forward(matches, **kwargs)
    second_goals, second_counts = run_walk_forward(matches, **kwargs)

    pd.testing.assert_frame_equal(first_goals, second_goals)
    pd.testing.assert_frame_equal(first_counts, second_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")
def test_training_and_evaluation_run_on_gpu_model():
    matches = _fake_matches(n_seasons=2)
    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    model = model.to("cuda")
    loader = make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=32)

    train_epochs(model, loader, torch.optim.Adam(model.parameters(), lr=0.01), num_epochs=1)
    goal_results = evaluate_goals(model, matches, division_vocab, team_vocab, means, stds)
    count_results = evaluate_counts(model, matches, division_vocab, team_vocab, means, stds)

    assert np.isfinite(goal_results["model_log_loss"]).all()
    assert np.isfinite(count_results["shots_on_target_nll"]).all()


def test_run_walk_forward_moves_model_to_requested_device(tmp_path):
    matches = _fake_matches(n_seasons=4)
    run_walk_forward(
        matches, burn_in_seasons=3, initial_epochs=1, fine_tune_epochs=1, checkpoint_dir=tmp_path, device="cpu"
    )
    checkpoint = torch.load(next(tmp_path.glob("multitask_*.pt")), weights_only=True)
    assert all(t.device.type == "cpu" for t in checkpoint["model_state_dict"].values())


def test_run_walk_forward_saves_per_match_results_next_to_checkpoint(tmp_path):
    matches = _fake_matches(n_seasons=5)
    goal_results, count_results = run_walk_forward(
        matches, burn_in_seasons=3, initial_epochs=1, fine_tune_epochs=1, checkpoint_dir=tmp_path
    )

    checkpoint = next(tmp_path.glob("multitask_*.pt"))
    goals_csv = tmp_path / f"{checkpoint.stem}_goals.csv"
    counts_csv = tmp_path / f"{checkpoint.stem}_counts.csv"
    assert goals_csv.exists() and counts_csv.exists()

    saved_goals = pd.read_csv(goals_csv)
    saved_counts = pd.read_csv(counts_csv)
    assert len(saved_goals) == len(goal_results)
    assert len(saved_counts) == len(count_results)
    assert set(saved_goals["Season"]) == {"2018-2019", "2019-2020"}  # only evaluation seasons
    assert np.allclose(saved_goals["model_log_loss"], goal_results["model_log_loss"])
