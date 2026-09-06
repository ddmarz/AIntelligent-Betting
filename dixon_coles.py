from dataclasses import dataclass


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
