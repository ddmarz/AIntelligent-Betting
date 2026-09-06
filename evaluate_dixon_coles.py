import math

import pandas as pd

from dixon_coles import fit_dixon_coles, predict_score_matrix, match_outcome_probs

TARGET_LEAGUES = ["I1", "E0", "SP1", "D1", "F1"]
BURN_IN_SEASONS = 3
XI_CANDIDATES = [0, 0.0005, 0.001, 0.002, 0.005]

COLUMNS = [
    "Division", "MatchDate", "Season", "HomeTeam", "AwayTeam",
    "FTHome", "FTAway", "FTResult", "OddHome", "OddDraw", "OddAway",
]


def load_target_matches(path="matches_clean.csv"):
    matches = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    matches["MatchDate"] = pd.to_datetime(matches["MatchDate"])
    matches = matches[matches["Division"].isin(TARGET_LEAGUES)]
    return matches.sort_values("MatchDate").reset_index(drop=True)


def implied_probabilities(odd_home, odd_draw, odd_away):
    inv_home, inv_draw, inv_away = 1 / odd_home, 1 / odd_draw, 1 / odd_away
    total = inv_home + inv_draw + inv_away
    return inv_home / total, inv_draw / total, inv_away / total


OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def log_loss(p_home, p_draw, p_away, outcome):
    probs = (p_home, p_draw, p_away)
    p_true = probs[OUTCOME_INDEX[outcome]]
    return -math.log(max(p_true, 1e-15))
