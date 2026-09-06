from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DixonColesModel:
    attack: dict
    defense: dict
    home_advantage: float
    rho: float
    teams: list


def tau_correction(x, y, lambda_home, lambda_away, rho):
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    elif x == 0 and y == 1:
        return 1 + lambda_home * rho
    elif x == 1 and y == 0:
        return 1 + lambda_away * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles(matches, as_of_date, xi):
    matches = matches[matches["MatchDate"] < as_of_date]
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    n = len(teams)
    idx = {team: i for i, team in enumerate(teams)}

    days = (as_of_date - matches["MatchDate"]).dt.days.to_numpy()
    weights = np.exp(-xi * days)
    home_idx = matches["HomeTeam"].map(idx).to_numpy()
    away_idx = matches["AwayTeam"].map(idx).to_numpy()
    ft_home = matches["FTHome"].to_numpy()
    ft_away = matches["FTAway"].to_numpy()

    def unpack(params):
        attack = params[:n]
        defense = params[n:2 * n]
        home_advantage = params[2 * n]
        rho = params[2 * n + 1]
        return attack, defense, home_advantage, rho

    def neg_log_likelihood(params):
        attack, defense, home_advantage, rho = unpack(params)
        lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
        lambda_away = np.exp(attack[away_idx] + defense[home_idx])

        log_p = poisson.logpmf(ft_home, lambda_home) + poisson.logpmf(ft_away, lambda_away)

        tau = np.ones(len(matches))
        low_score = (ft_home <= 1) & (ft_away <= 1)
        for i in np.where(low_score)[0]:
            tau[i] = tau_correction(ft_home[i], ft_away[i], lambda_home[i], lambda_away[i], rho)
        tau = np.clip(tau, 1e-10, None)

        return -np.sum(weights * (log_p + np.log(tau)))

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.1
    x0[2 * n + 1] = -0.1

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
    attack, defense, home_advantage, rho = unpack(result.x)

    c = attack.mean()
    attack = attack - c
    defense = defense + c

    return DixonColesModel(
        attack=dict(zip(teams, attack)),
        defense=dict(zip(teams, defense)),
        home_advantage=float(home_advantage),
        rho=float(rho),
        teams=teams,
    )
