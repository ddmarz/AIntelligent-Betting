import math

import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson
from torch.utils.data import DataLoader

from dl_model.data import MatchDataset, move_batch_to_device
from dl_model.losses import negative_binomial_nll


def poisson_score_matrix(lambda_home, lambda_away, max_goals=10):
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    matrix = np.outer(p_home, p_away)
    return matrix / matrix.sum()


def match_outcome_probs(score_matrix):
    p_home_win = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away_win = np.triu(score_matrix, 1).sum()
    return float(p_home_win), float(p_draw), float(p_away_win)


def implied_probabilities(odd_home, odd_draw, odd_away):
    inv_home, inv_draw, inv_away = 1 / odd_home, 1 / odd_draw, 1 / odd_away
    total = inv_home + inv_draw + inv_away
    return inv_home / total, inv_draw / total, inv_away / total


OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def log_loss(p_home, p_draw, p_away, outcome):
    probs = (p_home, p_draw, p_away)
    p_true = probs[OUTCOME_INDEX[outcome]]
    return -math.log(max(p_true, 1e-15))


MATCH_ID_COLUMNS = ["Division", "Season", "MatchDate", "HomeTeam", "AwayTeam"]
MATCH_ODDS_COLUMNS = ["OddHome", "OddDraw", "OddAway", "Over25", "Under25"]
GOAL_RESULT_COLUMNS = MATCH_ID_COLUMNS + [
    "FTResult", "FTHome", "FTAway", *MATCH_ODDS_COLUMNS,
    "lambda_home", "lambda_away",
    "model_p_home", "model_p_draw", "model_p_away",
    "market_p_home", "market_p_draw", "market_p_away",
    "model_log_loss", "market_log_loss",
]
COUNT_RESULT_COLUMNS = MATCH_ID_COLUMNS + [
    "HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners",
    "pred_home_shots_on_target", "pred_away_shots_on_target", "pred_home_corners", "pred_away_corners",
    "disp_home_shots_on_target", "disp_away_shots_on_target", "disp_home_corners", "disp_away_corners",
    "shots_on_target_nll", "corners_nll",
]


def evaluate_goals(model, matches, division_vocab, team_vocab, means, stds, batch_size=512):
    """One row per match: identity, model and market 1X2 probabilities, and their log-losses."""
    model.eval()
    match_rows = matches.reset_index(drop=True)
    dataset = MatchDataset(match_rows, division_vocab, team_vocab, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = next(model.parameters()).device
    records = []
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            continuous, division_idx, home_idx, away_idx, _targets = move_batch_to_device(batch, device)
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
                    **{col: row[col] for col in MATCH_ID_COLUMNS},
                    "FTResult": row["FTResult"],
                    "FTHome": row["FTHome"],
                    "FTAway": row["FTAway"],
                    **{col: row[col] for col in MATCH_ODDS_COLUMNS},
                    "lambda_home": lambda_home,
                    "lambda_away": lambda_away,
                    "model_p_home": p_home,
                    "model_p_draw": p_draw,
                    "model_p_away": p_away,
                    "market_p_home": m_home,
                    "market_p_draw": m_draw,
                    "market_p_away": m_away,
                    "model_log_loss": model_ll,
                    "market_log_loss": market_ll,
                })

            row_offset += batch_len

    return pd.DataFrame(records, columns=GOAL_RESULT_COLUMNS)


def evaluate_counts(model, matches, division_vocab, team_vocab, means, stds, batch_size=512):
    """One row per match: identity, actual and predicted shots on target/corners, and their NB-NLL."""
    model.eval()
    match_rows = matches.reset_index(drop=True)
    dataset = MatchDataset(match_rows, division_vocab, team_vocab, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = next(model.parameters()).device
    records = []
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            continuous, division_idx, home_idx, away_idx, targets = move_batch_to_device(batch, device)
            outputs = model(continuous, division_idx, home_idx, away_idx)

            shots_on_target_nll = negative_binomial_nll(
                outputs["mean_home_shots_on_target"], outputs["disp_home_shots_on_target"],
                targets["shots_on_target_home"],
            ) + negative_binomial_nll(
                outputs["mean_away_shots_on_target"], outputs["disp_away_shots_on_target"],
                targets["shots_on_target_away"],
            )
            corners_nll = negative_binomial_nll(
                outputs["mean_home_corners"], outputs["disp_home_corners"], targets["corners_home"]
            ) + negative_binomial_nll(
                outputs["mean_away_corners"], outputs["disp_away_corners"], targets["corners_away"]
            )

            batch_len = continuous.shape[0]
            for i in range(batch_len):
                row = match_rows.iloc[row_offset + i]
                records.append({
                    **{col: row[col] for col in MATCH_ID_COLUMNS},
                    "HomeTarget": row["HomeTarget"],
                    "AwayTarget": row["AwayTarget"],
                    "HomeCorners": row["HomeCorners"],
                    "AwayCorners": row["AwayCorners"],
                    "pred_home_shots_on_target": outputs["mean_home_shots_on_target"][i].item(),
                    "pred_away_shots_on_target": outputs["mean_away_shots_on_target"][i].item(),
                    "pred_home_corners": outputs["mean_home_corners"][i].item(),
                    "pred_away_corners": outputs["mean_away_corners"][i].item(),
                    "disp_home_shots_on_target": outputs["disp_home_shots_on_target"][i].item(),
                    "disp_away_shots_on_target": outputs["disp_away_shots_on_target"][i].item(),
                    "disp_home_corners": outputs["disp_home_corners"][i].item(),
                    "disp_away_corners": outputs["disp_away_corners"][i].item(),
                    "shots_on_target_nll": shots_on_target_nll[i].item(),
                    "corners_nll": corners_nll[i].item(),
                })

            row_offset += batch_len

    return pd.DataFrame(records, columns=COUNT_RESULT_COLUMNS)
