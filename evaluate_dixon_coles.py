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


def walk_forward_evaluate(league_matches, xi):
    seasons_in_order = list(dict.fromkeys(league_matches["Season"]))
    eval_seasons = seasons_in_order[BURN_IN_SEASONS:]

    records = []
    for season in eval_seasons:
        season_matches = league_matches[league_matches["Season"] == season]
        as_of_date = season_matches["MatchDate"].min()
        model = fit_dixon_coles(league_matches, as_of_date, xi)

        for _, row in season_matches.iterrows():
            if row["HomeTeam"] not in model.attack or row["AwayTeam"] not in model.attack:
                continue

            score_matrix = predict_score_matrix(model, row["HomeTeam"], row["AwayTeam"])
            p_home, p_draw, p_away = match_outcome_probs(score_matrix)
            model_ll = log_loss(p_home, p_draw, p_away, row["FTResult"])

            m_home, m_draw, m_away = implied_probabilities(row["OddHome"], row["OddDraw"], row["OddAway"])
            market_ll = log_loss(m_home, m_draw, m_away, row["FTResult"])

            records.append({
                "MatchDate": row["MatchDate"],
                "model_log_loss": model_ll,
                "market_log_loss": market_ll,
            })

    return pd.DataFrame(records)


def run_xi_grid_search(matches):
    rows = []
    for xi in XI_CANDIDATES:
        per_match = []
        for division in TARGET_LEAGUES:
            league_matches = matches[matches["Division"] == division]
            if league_matches.empty:
                continue
            per_match.append(walk_forward_evaluate(league_matches, xi))
        combined = pd.concat(per_match, ignore_index=True)
        rows.append({
            "xi": xi,
            "model_log_loss": combined["model_log_loss"].mean(),
            "market_log_loss": combined["market_log_loss"].mean(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    matches = load_target_matches()
    report = run_xi_grid_search(matches)
    print(report.to_string(index=False))
    best_xi = report.loc[report["model_log_loss"].idxmin(), "xi"]
    print(f"\nBest xi: {best_xi}")
