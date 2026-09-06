import torch
from dl_model.data import NUM_CONTINUOUS_FEATURES
from dl_model.model import MultiTaskMatchModel


def test_model_forward_produces_expected_output_shapes_and_positivity():
    model = MultiTaskMatchModel(num_divisions=5, num_teams=20)
    batch_size = 4
    continuous = torch.randn(batch_size, NUM_CONTINUOUS_FEATURES)
    division_idx = torch.randint(0, 5, (batch_size,))
    home_idx = torch.randint(0, 20, (batch_size,))
    away_idx = torch.randint(0, 20, (batch_size,))

    outputs = model(continuous, division_idx, home_idx, away_idx)

    expected_keys = {
        "lambda_home_goals", "lambda_away_goals",
        "mean_home_shots", "mean_away_shots", "disp_home_shots", "disp_away_shots",
        "mean_home_corners", "mean_away_corners", "disp_home_corners", "disp_away_corners",
    }
    assert set(outputs.keys()) == expected_keys
    for value in outputs.values():
        assert value.shape == (batch_size,)
        assert (value > 0).all()


def test_model_forward_works_with_batch_size_one():
    # BatchNorm1d needs eval() mode with a single sample (it cannot compute
    # batch statistics from one row) -- this matches real inference usage,
    # where evaluate_goals/evaluate_counts always call model.eval() first.
    model = MultiTaskMatchModel(num_divisions=5, num_teams=20)
    model.eval()
    continuous = torch.randn(1, NUM_CONTINUOUS_FEATURES)
    division_idx = torch.randint(0, 5, (1,))
    home_idx = torch.randint(0, 20, (1,))
    away_idx = torch.randint(0, 20, (1,))

    outputs = model(continuous, division_idx, home_idx, away_idx)

    assert outputs["lambda_home_goals"].shape == (1,)
