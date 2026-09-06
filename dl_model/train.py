import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_dixon_coles import TARGET_LEAGUES
from dl_model.data import build_vocab, compute_normalization_stats, MatchDataset, load_all_matches
from dl_model.model import MultiTaskMatchModel
from dl_model.losses import poisson_nll, negative_binomial_nll, masked_mean
from dl_model.evaluation import evaluate_goals, evaluate_counts

BURN_IN_SEASONS = 5
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.0005


def build_model_and_vocabs(matches):
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(matches)
    model = MultiTaskMatchModel(num_divisions=len(division_vocab), num_teams=len(team_vocab))
    return model, division_vocab, team_vocab, means, stds


def make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=BATCH_SIZE,
                     shuffle=True, drop_last=False):
    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def compute_total_loss(outputs, targets):
    goals_loss = (
        poisson_nll(outputs["lambda_home_goals"], targets["goals_home"]).mean()
        + poisson_nll(outputs["lambda_away_goals"], targets["goals_away"]).mean()
    )

    shots_loss_elementwise = (
        negative_binomial_nll(outputs["mean_home_shots"], outputs["disp_home_shots"], targets["shots_home"])
        + negative_binomial_nll(outputs["mean_away_shots"], outputs["disp_away_shots"], targets["shots_away"])
    )
    shots_loss = masked_mean(shots_loss_elementwise, targets["has_stats"])

    corners_loss_elementwise = (
        negative_binomial_nll(outputs["mean_home_corners"], outputs["disp_home_corners"], targets["corners_home"])
        + negative_binomial_nll(outputs["mean_away_corners"], outputs["disp_away_corners"], targets["corners_away"])
    )
    corners_loss = masked_mean(corners_loss_elementwise, targets["has_stats"])

    return goals_loss + shots_loss + corners_loss


def train_epochs(model, dataloader, optimizer, num_epochs):
    model.train()
    for _epoch in range(num_epochs):
        for continuous, division_idx, home_idx, away_idx, targets in dataloader:
            if continuous.shape[0] < 2:
                continue  # BatchNorm1d cannot compute batch statistics from a single sample

            optimizer.zero_grad()
            outputs = model(continuous, division_idx, home_idx, away_idx)
            loss = compute_total_loss(outputs, targets)
            loss.backward()
            optimizer.step()


def run_walk_forward(matches, burn_in_seasons=BURN_IN_SEASONS, initial_epochs=INITIAL_EPOCHS,
                      fine_tune_epochs=FINE_TUNE_EPOCHS):
    matches = matches.sort_values("MatchDate").reset_index(drop=True)
    seasons_in_order = list(dict.fromkeys(matches["Season"]))
    burn_in_season_labels = seasons_in_order[:burn_in_seasons]
    eval_seasons = seasons_in_order[burn_in_seasons:]

    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    burn_in_matches = matches[matches["Season"].isin(burn_in_season_labels)]
    burn_in_loader = make_dataloader(burn_in_matches, division_vocab, team_vocab, means, stds, drop_last=True)
    train_epochs(model, burn_in_loader, optimizer, initial_epochs)

    all_goal_results = []
    all_count_results = []

    for season in eval_seasons:
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
        train_epochs(model, fine_tune_loader, optimizer, fine_tune_epochs)

    goal_results = pd.concat(all_goal_results, ignore_index=True) if all_goal_results else pd.DataFrame(
        columns=["MatchDate", "model_log_loss", "market_log_loss"]
    )
    count_results = pd.concat(all_count_results, ignore_index=True) if all_count_results else pd.DataFrame(
        columns=["shots_nll", "corners_nll"]
    )
    return goal_results, count_results


if __name__ == "__main__":
    matches = load_all_matches()
    goal_results, count_results = run_walk_forward(matches)

    print("Goals task (Big 5, model vs market):")
    print(f"  model log-loss:  {goal_results['model_log_loss'].mean():.4f}")
    print(f"  market log-loss: {goal_results['market_log_loss'].mean():.4f}")

    print("\nShots/corners task (has_stats rows, Negative-Binomial NLL):")
    print(f"  shots NLL:   {count_results['shots_nll'].mean():.4f}")
    print(f"  corners NLL: {count_results['corners_nll'].mean():.4f}")
