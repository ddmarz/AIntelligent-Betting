import numpy as np
import pandas as pd
import torch
from dl_model.data import CONTINUOUS_COLUMNS
from dl_model.train import (
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
                        "OddHome": 2.0, "OddDraw": 3.2, "OddAway": 3.8,
                        "HomeShots": sh, "AwayShots": sa, "HomeCorners": ch, "AwayCorners": ca,
                        "has_stats": True,
                    }
                    for col in CONTINUOUS_COLUMNS:
                        row[col] = float(np.random.randn())
                    rows.append(row)
    return pd.DataFrame(rows).sort_values("MatchDate").reset_index(drop=True)


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


def test_run_walk_forward_returns_expected_report_shapes():
    matches = _fake_matches(n_seasons=6)

    goal_results, count_results = run_walk_forward(
        matches, burn_in_seasons=3, initial_epochs=2, fine_tune_epochs=1
    )

    assert set(goal_results.columns) == {"MatchDate", "model_log_loss", "market_log_loss"}
    assert set(count_results.columns) == {"shots_nll", "corners_nll"}
    assert not goal_results.empty
    assert not count_results.empty
    assert np.isfinite(goal_results["model_log_loss"]).all()
