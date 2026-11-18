import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_model.data import (
    TARGET_LEAGUES,
    build_vocab,
    compute_normalization_stats,
    MatchDataset,
    load_all_matches,
    move_batch_to_device,
)
from dl_model.model import MultiTaskMatchModel
from dl_model.losses import poisson_nll, negative_binomial_nll, masked_mean
from dl_model.evaluation import evaluate_goals, evaluate_counts, GOAL_RESULT_COLUMNS, COUNT_RESULT_COLUMNS

BURN_IN_SEASONS = 5
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
SEED = 1926
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_and_vocabs(matches, normalization_matches=None):
    """
        Vocabularies are built on every match (team/division names only).
        Normalization stats come from normalization_matches (defaults to matches): in the
        walk-forward these are the burn-in seasons, so no statistic leaks from evaluation seasons.
    """
    if normalization_matches is None:
        normalization_matches = matches
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(normalization_matches)
    model = MultiTaskMatchModel(num_divisions=len(division_vocab), num_teams=len(team_vocab))
    return model, division_vocab, team_vocab, means, stds


def save_checkpoint(model, division_vocab, team_vocab, means, stds, config, checkpoint_dir=CHECKPOINT_DIR):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"multitask_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.pt"
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "division_vocab": division_vocab,
        "team_vocab": team_vocab,
        "means": {key: float(value) for key, value in means.items()},
        "stds": {key: float(value) for key, value in stds.items()},
        **config,
    }, path)
    return path


def save_results(goal_results, count_results, checkpoint_path):
    """Save per-match evaluation results next to the checkpoint, sharing its name."""
    checkpoint_path = Path(checkpoint_path)
    goals_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_goals.csv")
    counts_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_counts.csv")
    goal_results.to_csv(goals_path, index=False)
    count_results.to_csv(counts_path, index=False)
    return goals_path, counts_path


def make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=BATCH_SIZE,
                     shuffle=True, drop_last=False):
    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def compute_total_loss(outputs, targets):
    goals_loss = (
        poisson_nll(outputs["lambda_home_goals"], targets["goals_home"]).mean()
        + poisson_nll(outputs["lambda_away_goals"], targets["goals_away"]).mean()
    )

    shots_on_target_loss_elementwise = (
        negative_binomial_nll(
            outputs["mean_home_shots_on_target"], outputs["disp_home_shots_on_target"], targets["shots_on_target_home"]
        )
        + negative_binomial_nll(
            outputs["mean_away_shots_on_target"], outputs["disp_away_shots_on_target"], targets["shots_on_target_away"]
        )
    )
    shots_on_target_loss = masked_mean(shots_on_target_loss_elementwise, targets["has_stats"])

    corners_loss_elementwise = (
        negative_binomial_nll(outputs["mean_home_corners"], outputs["disp_home_corners"], targets["corners_home"])
        + negative_binomial_nll(outputs["mean_away_corners"], outputs["disp_away_corners"], targets["corners_away"])
    )
    corners_loss = masked_mean(corners_loss_elementwise, targets["has_stats"])

    return goals_loss + shots_on_target_loss + corners_loss


def train_epochs(model, dataloader, optimizer, num_epochs, desc="Train"):
    """
        Trains for num_epochs and returns the mean batch loss of each epoch.
        Shows a progress bar over the batches of each epoch (stderr) and logs one line
        per epoch with its mean loss (stdout).
    """
    model.train()
    device = next(model.parameters()).device
    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        epoch_label = f"{desc} | epoch {epoch}/{num_epochs}"
        batch_losses = []
        progress = tqdm(dataloader, desc=epoch_label, unit="batch", leave=False)
        for batch in progress:
            continuous, division_idx, home_idx, away_idx, targets = move_batch_to_device(batch, device)
            if continuous.shape[0] < 2:
                continue  # BatchNorm1d cannot compute batch statistics from a single sample

            optimizer.zero_grad()
            outputs = model(continuous, division_idx, home_idx, away_idx)
            loss = compute_total_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        epoch_losses.append(epoch_loss)
        tqdm.write(f"{epoch_label} | loss {epoch_loss:.4f}", file=sys.stdout)
    return epoch_losses


def run_walk_forward(matches, burn_in_seasons=BURN_IN_SEASONS, initial_epochs=INITIAL_EPOCHS,
                      fine_tune_epochs=FINE_TUNE_EPOCHS, seed=SEED, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR):
    set_seed(seed)

    matches = matches.sort_values("MatchDate").reset_index(drop=True)

    # Takes the seasons and identifies the seasons (and matches) for the burn-in.
    seasons_in_order = list(dict.fromkeys(matches["Season"]))
    burn_in_season_labels = seasons_in_order[:burn_in_seasons]
    eval_seasons = seasons_in_order[burn_in_seasons:]
    burn_in_matches = matches[matches["Season"].isin(burn_in_season_labels)]

    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(
        matches, normalization_matches=burn_in_matches
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    burn_in_loader = make_dataloader(burn_in_matches, division_vocab, team_vocab, means, stds, drop_last=True)
    train_epochs(model, burn_in_loader, optimizer, initial_epochs, desc="Burn-in")

    all_goal_results = []
    all_count_results = []

    for season_number, season in enumerate(eval_seasons, start=1):
        season_matches = matches[matches["Season"] == season]

        goal_eval_matches = season_matches[season_matches["Division"].isin(TARGET_LEAGUES)]
        if not goal_eval_matches.empty:
            all_goal_results.append(
                evaluate_goals(model, goal_eval_matches, division_vocab, team_vocab, means, stds)
            )

        count_eval_matches = season_matches[season_matches["has_stats"]]
        if not count_eval_matches.empty:
            all_count_results.append(
                evaluate_counts(model, count_eval_matches, division_vocab, team_vocab, means, stds)
            )

        fine_tune_loader = make_dataloader(
            season_matches, division_vocab, team_vocab, means, stds, drop_last=True
        )
        train_epochs(
            model, fine_tune_loader, optimizer, fine_tune_epochs,
            desc=f"Season {season} ({season_number}/{len(eval_seasons)})",
        )

    goal_results = pd.concat(all_goal_results, ignore_index=True) if all_goal_results else pd.DataFrame(
        columns=GOAL_RESULT_COLUMNS
    )
    count_results = pd.concat(all_count_results, ignore_index=True) if all_count_results else pd.DataFrame(
        columns=COUNT_RESULT_COLUMNS
    )

    config = {
        "seed": seed,
        "burn_in_seasons": burn_in_seasons,
        "initial_epochs": initial_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "last_season": seasons_in_order[-1],
    }
    checkpoint_path = save_checkpoint(model, division_vocab, team_vocab, means, stds, config, checkpoint_dir)
    print(f"Checkpoint saved to {checkpoint_path}")
    goals_path, counts_path = save_results(goal_results, count_results, checkpoint_path)
    print(f"Per-match results saved to {goals_path} and {counts_path}")

    return goal_results, count_results


if __name__ == "__main__":

    # Loading
    matches = load_all_matches()


    goal_results, count_results = run_walk_forward(matches)

    print("Goals task (Big 5, model vs market):")
    print(f"  model log-loss:  {goal_results['model_log_loss'].mean():.4f}")
    print(f"  market log-loss: {goal_results['market_log_loss'].mean():.4f}")

    print("\nShots-on-target/corners task (has_stats rows, Negative-Binomial NLL):")
    print(f"  shots on target NLL: {count_results['shots_on_target_nll'].mean():.4f}")
    print(f"  corners NLL:         {count_results['corners_nll'].mean():.4f}")
