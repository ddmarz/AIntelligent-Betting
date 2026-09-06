from dixon_coles import DixonColesModel, tau_correction


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
