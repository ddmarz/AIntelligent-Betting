import torch


def poisson_nll(lam, target):
    return lam - target * torch.log(lam)


def negative_binomial_nll(mean, dispersion, target):
    eps = 1e-8
    mean = mean.clamp(min=eps)
    r = dispersion.clamp(min=eps)
    log_prob = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1)
        + r * torch.log(r / (r + mean))
        + target * torch.log(mean / (r + mean))
    )
    return -log_prob


def masked_mean(loss, mask):
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)
