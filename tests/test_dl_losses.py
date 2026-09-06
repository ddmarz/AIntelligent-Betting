import torch
from scipy.stats import nbinom
from dl_model.losses import poisson_nll, negative_binomial_nll, masked_mean


def test_poisson_nll_matches_manual_formula():
    lam = torch.tensor([2.0, 1.5])
    target = torch.tensor([3.0, 0.0])
    result = poisson_nll(lam, target)
    expected = lam - target * torch.log(lam)
    assert torch.allclose(result, expected)


def test_poisson_nll_is_lower_when_lambda_matches_target():
    target = torch.tensor([4.0])
    loss_at_target = poisson_nll(torch.tensor([4.0]), target)
    loss_away_from_target = poisson_nll(torch.tensor([1.0]), target)
    assert loss_at_target.item() < loss_away_from_target.item()


def test_negative_binomial_nll_matches_scipy():
    mean, dispersion, target = 5.0, 2.0, 3.0
    p = dispersion / (dispersion + mean)
    expected = -nbinom.logpmf(target, dispersion, p)

    result = negative_binomial_nll(
        torch.tensor([mean]), torch.tensor([dispersion]), torch.tensor([target])
    )
    assert abs(result.item() - expected) < 1e-4


def test_masked_mean_ignores_masked_entries():
    loss = torch.tensor([10.0, 20.0, 30.0])
    mask = torch.tensor([True, False, True])
    result = masked_mean(loss, mask)
    assert abs(result.item() - 20.0) < 1e-6
