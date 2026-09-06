import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson
from torch.utils.data import DataLoader

from dixon_coles import match_outcome_probs
from evaluate_dixon_coles import implied_probabilities, log_loss
from dl_model.data import MatchDataset
from dl_model.losses import negative_binomial_nll


def poisson_score_matrix(lambda_home, lambda_away, max_goals=10):
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    matrix = np.outer(p_home, p_away)
    return matrix / matrix.sum()


def evaluate_goals(model, matches, division_vocab, team_vocab, means, stds, batch_size=512):
    model.eval()
    match_rows = matches.reset_index(drop=True)
    dataset = MatchDataset(match_rows, division_vocab, team_vocab, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records = []
    row_offset = 0
    with torch.no_grad():
        for continuous, division_idx, home_idx, away_idx, _targets in loader:
            outputs = model(continuous, division_idx, home_idx, away_idx)
            batch_len = continuous.shape[0]

            for i in range(batch_len):
                row = match_rows.iloc[row_offset + i]
                lambda_home = outputs["lambda_home_goals"][i].item()
                lambda_away = outputs["lambda_away_goals"][i].item()

                score_matrix = poisson_score_matrix(lambda_home, lambda_away)
                p_home, p_draw, p_away = match_outcome_probs(score_matrix)
                model_ll = log_loss(p_home, p_draw, p_away, row["FTResult"])

                m_home, m_draw, m_away = implied_probabilities(row["OddHome"], row["OddDraw"], row["OddAway"])
                market_ll = log_loss(m_home, m_draw, m_away, row["FTResult"])

                records.append({
                    "MatchDate": row["MatchDate"],
                    "model_log_loss": model_ll,
                    "market_log_loss": market_ll,
                })

            row_offset += batch_len

    return pd.DataFrame(records)


def evaluate_counts(model, matches, division_vocab, team_vocab, means, stds, batch_size=512):
    model.eval()
    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records = []
    with torch.no_grad():
        for continuous, division_idx, home_idx, away_idx, targets in loader:
            outputs = model(continuous, division_idx, home_idx, away_idx)

            shots_nll = negative_binomial_nll(
                outputs["mean_home_shots"], outputs["disp_home_shots"], targets["shots_home"]
            ) + negative_binomial_nll(
                outputs["mean_away_shots"], outputs["disp_away_shots"], targets["shots_away"]
            )
            corners_nll = negative_binomial_nll(
                outputs["mean_home_corners"], outputs["disp_home_corners"], targets["corners_home"]
            ) + negative_binomial_nll(
                outputs["mean_away_corners"], outputs["disp_away_corners"], targets["corners_away"]
            )

            for i in range(continuous.shape[0]):
                records.append({
                    "shots_nll": shots_nll[i].item(),
                    "corners_nll": corners_nll[i].item(),
                })

    return pd.DataFrame(records)
