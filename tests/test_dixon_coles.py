import numpy as np
import pandas as pd
from dixon_coles import DixonColesModel, tau_correction, fit_dixon_coles


def test_dixon_coles_model_holds_fields():
    model = DixonColesModel(
        attack={"A": 0.1, "B": -0.1},
        defense={"A": -0.05, "B": 0.05},
        home_advantage=0.3,
        rho=-0.1,
        teams=["A", "B"],
    )
    assert model.attack["A"] == 0.1
    assert model.teams == ["A", "B"]


def test_tau_correction_low_scores():
    lam_h, lam_a, rho = 1.5, 1.2, -0.15
    assert tau_correction(0, 0, lam_h, lam_a, rho) == 1 - lam_h * lam_a * rho
    assert tau_correction(0, 1, lam_h, lam_a, rho) == 1 + lam_h * rho
    assert tau_correction(1, 0, lam_h, lam_a, rho) == 1 + lam_a * rho
    assert tau_correction(1, 1, lam_h, lam_a, rho) == 1 - rho


def test_tau_correction_other_scores_untouched():
    assert tau_correction(2, 0, 1.5, 1.2, -0.15) == 1.0
    assert tau_correction(0, 2, 1.5, 1.2, -0.15) == 1.0
    assert tau_correction(3, 4, 1.5, 1.2, -0.15) == 1.0


def _synthetic_matches():
    np.random.seed(42)
    teams = ["Strong", "Good", "Mid", "Weak"]
    true_attack = {"Strong": 0.6, "Good": 0.2, "Mid": 0.0, "Weak": -0.5}
    true_defense = {"Strong": -0.3, "Good": -0.1, "Mid": 0.0, "Weak": 0.3}
    home_adv = 0.25

    rows = []
    start = pd.Timestamp("2020-08-01")
    for round_num in range(60):
        date = start + pd.Timedelta(days=round_num * 3)
        shuffled = teams[:]
        np.random.shuffle(shuffled)
        for i in range(0, len(shuffled), 2):
            h, a = shuffled[i], shuffled[i + 1]
            lam_h = np.exp(true_attack[h] + true_defense[a] + home_adv)
            lam_a = np.exp(true_attack[a] + true_defense[h])
            gh = np.random.poisson(lam_h)
            ga = np.random.poisson(lam_a)
            rows.append({"MatchDate": date, "HomeTeam": h, "AwayTeam": a, "FTHome": gh, "FTAway": ga})
    return pd.DataFrame(rows)


def test_fit_dixon_coles_recovers_team_ranking():
    matches = _synthetic_matches()
    as_of = matches["MatchDate"].max() + pd.Timedelta(days=7)

    model = fit_dixon_coles(matches, as_of, xi=0.001)

    assert abs(sum(model.attack.values())) < 1e-8
    assert model.attack["Strong"] > model.attack["Good"] > model.attack["Mid"] > model.attack["Weak"]
    assert set(model.teams) == {"Strong", "Good", "Mid", "Weak"}


def test_fit_dixon_coles_ignores_matches_after_cutoff():
    matches = _synthetic_matches()
    cutoff = matches["MatchDate"].min() + pd.Timedelta(days=10)

    model = fit_dixon_coles(matches, cutoff, xi=0.001)

    used_teams = set(matches[matches["MatchDate"] < cutoff]["HomeTeam"]) | set(
        matches[matches["MatchDate"] < cutoff]["AwayTeam"]
    )
    assert set(model.teams) == used_teams
