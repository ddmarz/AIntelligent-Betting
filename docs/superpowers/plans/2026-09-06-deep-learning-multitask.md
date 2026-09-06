# Deep Learning Multi-Task Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note for this project:** PyTorch is not installed in this environment. All code and tests in this plan are written to be correct and runnable once PyTorch is installed, but were NOT executed during authoring. The first thing to do after installing PyTorch is run the full test suite (`python -m pytest tests/ -v`) before trusting any of it.

**Goal:** A multi-task PyTorch model (CNN 1D + Transformer, adapted from Gao, Cheng & Gao 2025) that predicts goals, shots, and corners for both teams in a match, trained with walk-forward fine-tuning, evaluated against Dixon-Coles/market on goals and via Negative-Binomial NLL on shots/corners.

**Architecture:** A `dl_model/` package: `data.py` (vocab/normalization/Dataset), `model.py` (the network), `losses.py` (Poisson and Negative-Binomial NLL), `evaluation.py` (goal log-loss vs market, count NLL), `train.py` (training loop, walk-forward orchestration, report).

**Tech Stack:** Python, PyTorch, pandas, numpy, scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-deep-learning-multitask-design.md`

## Global Constraints

- Categorical inputs: `Division`, `HomeTeam`, `AwayTeam` (embeddings, unknown-token fallback for unseen values).
- Continuous inputs (19 columns, see `dl_model.data.CONTINUOUS_COLUMNS`): `EloDiff`, 4 rolling-goal columns, 4 form columns, 6 cluster columns, 4 `has_*` flags.
- No market odds as model input.
- Goals: Poisson heads. Shots/corners: Negative-Binomial heads (mean + dispersion), loss masked to `has_stats=True` rows only.
- Goals task trains/evaluates on all 38 leagues; comparison against Dixon-Coles/market restricted to `TARGET_LEAGUES` (`I1, E0, SP1, D1, F1`, imported from `evaluate_dixon_coles`).
- Training: full training on a burn-in window of seasons, then per-season fine-tuning walk-forward (not full retraining) — mirrors the cited paper's rolling-window approach.
- No model persistence beyond what the walk-forward loop needs internally.

---

### Task 1: Loss functions (`dl_model/losses.py`)

**Files:**
- Create: `dl_model/__init__.py` (empty)
- Create: `dl_model/losses.py`
- Test: `tests/test_dl_losses.py`

**Interfaces:**
- Produces: `poisson_nll(lam: Tensor, target: Tensor) -> Tensor` (elementwise, no reduction), `negative_binomial_nll(mean: Tensor, dispersion: Tensor, target: Tensor) -> Tensor` (elementwise, mean-dispersion parameterization), `masked_mean(loss: Tensor, mask: Tensor) -> Tensor` (mean over `loss` where `mask` is `True`, ignoring the rest).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dl_losses.py
import numpy as np
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dl_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dl_model'`

- [ ] **Step 3: Write the minimal implementation**

```python
# dl_model/__init__.py
```

```python
# dl_model/losses.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dl_losses.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add dl_model/__init__.py dl_model/losses.py tests/test_dl_losses.py
git commit -m "Add Poisson and Negative-Binomial NLL losses for multi-task model"
```

---

### Task 2: Data preparation (`dl_model/data.py`)

**Files:**
- Create: `dl_model/data.py`
- Test: `tests/test_dl_data.py`

**Interfaces:**
- Produces: `CONTINUOUS_COLUMNS: list[str]` (19 entries), `NUM_CONTINUOUS_FEATURES: int` (`len(CONTINUOUS_COLUMNS)`), `UNKNOWN_TOKEN: str`, `build_vocab(values: Iterable[str]) -> dict[str, int]` (assigns `UNKNOWN_TOKEN` to index 0, every other unique value to a distinct index), `encode_categorical(value: str, vocab: dict[str, int]) -> int` (falls back to the unknown index for values not in `vocab`), `compute_normalization_stats(matches: pandas.DataFrame, columns=CONTINUOUS_COLUMNS) -> tuple[pandas.Series, pandas.Series]` (means, stds — stds of 0 replaced with 1.0), `MatchDataset(torch.utils.data.Dataset)` constructed as `MatchDataset(matches, division_vocab, team_vocab, means, stds)`, `__getitem__` returns `(continuous: Tensor[19], division_idx: Tensor[], home_idx: Tensor[], away_idx: Tensor[], targets: dict)` where `targets` has keys `goals_home, goals_away, shots_home, shots_away, corners_home, corners_away` (float tensors) and `has_stats` (bool tensor), taken directly from the matching `matches_clean.csv` columns (`FTHome, FTAway, HomeShots, AwayShots, HomeCorners, AwayCorners, has_stats`) — these are already imputed/flagged by Fase 1's `dataset.py`, so no NaN handling is needed here. Also produces `load_all_matches(path: str = "matches_clean.csv") -> pandas.DataFrame` (reads the full clean dataset, no league filtering — unlike `evaluate_dixon_coles.load_target_matches`, this keeps all 38 leagues for the goals task — with `MatchDate` parsed as datetime, sorted by `MatchDate`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dl_data.py
import pandas as pd
import torch
from dl_model.data import (
    CONTINUOUS_COLUMNS,
    NUM_CONTINUOUS_FEATURES,
    UNKNOWN_TOKEN,
    build_vocab,
    encode_categorical,
    compute_normalization_stats,
    MatchDataset,
    load_all_matches,
)


def _sample_matches():
    data = {
        "Division": ["I1", "I1", "E0"],
        "MatchDate": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-02"]),
        "HomeTeam": ["Juventus", "Milan", "Arsenal"],
        "AwayTeam": ["Milan", "Roma", "Chelsea"],
        "FTHome": [2, 1, 0],
        "FTAway": [1, 1, 0],
        "FTResult": ["H", "D", "D"],
        "OddHome": [1.8, 2.5, 2.2],
        "OddDraw": [3.5, 3.2, 3.1],
        "OddAway": [4.0, 2.8, 3.4],
        "HomeShots": [15, 10, 8],
        "AwayShots": [9, 11, 7],
        "HomeCorners": [6, 4, 3],
        "AwayCorners": [3, 5, 2],
        "has_stats": [True, True, False],
    }
    for i, col in enumerate(CONTINUOUS_COLUMNS):
        data[col] = [0.1 + i * 0.01, -0.2 + i * 0.01, 0.0 + i * 0.01]
    return pd.DataFrame(data)


def test_build_vocab_assigns_unique_indices_with_unknown_at_zero():
    vocab = build_vocab(["Juventus", "Milan", "Juventus"])
    assert vocab[UNKNOWN_TOKEN] == 0
    assert len(vocab) == 3
    assert len(set(vocab.values())) == 3


def test_encode_categorical_falls_back_to_unknown():
    vocab = build_vocab(["Juventus", "Milan"])
    assert encode_categorical("NeverSeenTeam", vocab) == vocab[UNKNOWN_TOKEN]
    assert encode_categorical("Juventus", vocab) == vocab["Juventus"]


def test_compute_normalization_stats_returns_mean_and_std_per_column():
    matches = _sample_matches()
    means, stds = compute_normalization_stats(matches)
    assert set(means.index) == set(CONTINUOUS_COLUMNS)
    assert (stds > 0).all()


def test_match_dataset_returns_expected_tensor_shapes_and_values():
    matches = _sample_matches()
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(matches)

    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    assert len(dataset) == 3

    continuous, division_idx, home_idx, away_idx, targets = dataset[0]

    assert continuous.shape == (NUM_CONTINUOUS_FEATURES,)
    assert division_idx.dtype == torch.long
    assert home_idx.item() == team_vocab["Juventus"]
    assert away_idx.item() == team_vocab["Milan"]
    assert targets["goals_home"].item() == 2.0
    assert targets["goals_away"].item() == 1.0
    assert targets["has_stats"].item() is True


def test_load_all_matches_keeps_all_leagues(tmp_path):
    csv_path = tmp_path / "matches_clean.csv"
    _sample_matches().to_csv(csv_path, index=False)

    matches = load_all_matches(str(csv_path))

    assert set(matches["Division"]) == {"I1", "E0"}
    assert pd.api.types.is_datetime64_any_dtype(matches["MatchDate"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dl_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'CONTINUOUS_COLUMNS'`

- [ ] **Step 3: Write the minimal implementation**

```python
# dl_model/data.py
import pandas as pd
import torch
from torch.utils.data import Dataset

CONTINUOUS_COLUMNS = [
    "EloDiff",
    "HomeAvgGoalsFor5", "HomeAvgGoalsAgainst5", "AwayAvgGoalsFor5", "AwayAvgGoalsAgainst5",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB",
    "has_elo", "has_rolling_stats", "has_form", "has_cluster_data",
]
NUM_CONTINUOUS_FEATURES = len(CONTINUOUS_COLUMNS)

UNKNOWN_TOKEN = "<unknown>"


def build_vocab(values):
    unique_values = sorted(set(values))
    vocab = {UNKNOWN_TOKEN: 0}
    for value in unique_values:
        vocab[value] = len(vocab)
    return vocab


def encode_categorical(value, vocab):
    return vocab.get(value, vocab[UNKNOWN_TOKEN])


def compute_normalization_stats(matches, columns=CONTINUOUS_COLUMNS):
    means = matches[columns].astype(float).mean()
    stds = matches[columns].astype(float).std().replace(0, 1.0)
    return means, stds


class MatchDataset(Dataset):
    def __init__(self, matches, division_vocab, team_vocab, means, stds, columns=CONTINUOUS_COLUMNS):
        self.matches = matches.reset_index(drop=True)
        self.division_vocab = division_vocab
        self.team_vocab = team_vocab
        self.means = means
        self.stds = stds
        self.columns = columns

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        row = self.matches.iloc[idx]

        continuous = (row[self.columns].astype(float) - self.means) / self.stds
        continuous_tensor = torch.tensor(continuous.to_numpy(), dtype=torch.float32)

        division_idx = torch.tensor(encode_categorical(row["Division"], self.division_vocab), dtype=torch.long)
        home_idx = torch.tensor(encode_categorical(row["HomeTeam"], self.team_vocab), dtype=torch.long)
        away_idx = torch.tensor(encode_categorical(row["AwayTeam"], self.team_vocab), dtype=torch.long)

        targets = {
            "goals_home": torch.tensor(float(row["FTHome"]), dtype=torch.float32),
            "goals_away": torch.tensor(float(row["FTAway"]), dtype=torch.float32),
            "shots_home": torch.tensor(float(row["HomeShots"]), dtype=torch.float32),
            "shots_away": torch.tensor(float(row["AwayShots"]), dtype=torch.float32),
            "corners_home": torch.tensor(float(row["HomeCorners"]), dtype=torch.float32),
            "corners_away": torch.tensor(float(row["AwayCorners"]), dtype=torch.float32),
            "has_stats": torch.tensor(bool(row["has_stats"])),
        }

        return continuous_tensor, division_idx, home_idx, away_idx, targets


def load_all_matches(path="matches_clean.csv"):
    matches = pd.read_csv(path, low_memory=False)
    matches["MatchDate"] = pd.to_datetime(matches["MatchDate"])
    return matches.sort_values("MatchDate").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dl_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add dl_model/data.py tests/test_dl_data.py
git commit -m "Add vocab/normalization/Dataset for the multi-task model"
```

---

### Task 3: Model architecture (`dl_model/model.py`)

**Files:**
- Create: `dl_model/model.py`
- Test: `tests/test_dl_model.py`

**Interfaces:**
- Consumes: `NUM_CONTINUOUS_FEATURES` (from `dl_model.data`, Task 2).
- Produces: `MultiTaskMatchModel(torch.nn.Module)` constructed as `MultiTaskMatchModel(num_divisions: int, num_teams: int, conv_hidden: int = 16, team_dim: int = 16, division_dim: int = 8, fusion_dim: int = 64)`. `forward(continuous: Tensor[B, 19], division_idx: Tensor[B], home_team_idx: Tensor[B], away_team_idx: Tensor[B]) -> dict[str, Tensor[B]]` with exactly these keys, every value strictly positive: `lambda_home_goals, lambda_away_goals, mean_home_shots, mean_away_shots, disp_home_shots, disp_away_shots, mean_home_corners, mean_away_corners, disp_home_corners, disp_away_corners`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dl_model.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dl_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dl_model.model'`

- [ ] **Step 3: Write the minimal implementation**

```python
# dl_model/model.py
import torch
import torch.nn as nn

from dl_model.data import NUM_CONTINUOUS_FEATURES


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.activation = nn.LeakyReLU()

        pooled_len = seq_len // 2
        self.residual_fc = nn.Linear(in_channels * seq_len, hidden_channels * pooled_len)
        self.output_dim = hidden_channels * pooled_len

    def forward(self, x):
        batch_size = x.shape[0]
        residual = self.residual_fc(x.reshape(batch_size, -1))

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out.reshape(batch_size, -1)

        return out + residual


class CategoricalTransformerBranch(nn.Module):
    def __init__(self, num_divisions, num_teams, division_dim=8, team_dim=16, num_heads=4, ff_dim=64):
        super().__init__()
        self.division_embedding = nn.Embedding(num_divisions, division_dim)
        self.team_embedding = nn.Embedding(num_teams, team_dim)
        self.division_proj = nn.Linear(division_dim, team_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=team_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_dim = team_dim * 3

    def forward(self, division_idx, home_team_idx, away_team_idx):
        division_tok = self.division_proj(self.division_embedding(division_idx))
        home_tok = self.team_embedding(home_team_idx)
        away_tok = self.team_embedding(away_team_idx)

        tokens = torch.stack([division_tok, home_tok, away_tok], dim=1)
        encoded = self.encoder(tokens)
        return encoded.reshape(encoded.shape[0], -1)


class MultiTaskMatchModel(nn.Module):
    def __init__(self, num_divisions, num_teams, conv_hidden=16, team_dim=16, division_dim=8, fusion_dim=64):
        super().__init__()
        self.conv_branch = ResidualConvBlock(1, conv_hidden, NUM_CONTINUOUS_FEATURES)
        self.categorical_branch = CategoricalTransformerBranch(num_divisions, num_teams, division_dim, team_dim)

        fused_input_dim = self.conv_branch.output_dim + self.categorical_branch.output_dim
        self.fusion = nn.Sequential(nn.Linear(fused_input_dim, fusion_dim), nn.LeakyReLU())

        self.goals_head = nn.Linear(fusion_dim, 2)
        self.shots_head = nn.Linear(fusion_dim, 4)
        self.corners_head = nn.Linear(fusion_dim, 4)

    def forward(self, continuous, division_idx, home_team_idx, away_team_idx):
        conv_out = self.conv_branch(continuous.unsqueeze(1))
        cat_out = self.categorical_branch(division_idx, home_team_idx, away_team_idx)
        fused = self.fusion(torch.cat([conv_out, cat_out], dim=1))

        goals_raw = self.goals_head(fused)
        shots_raw = self.shots_head(fused)
        corners_raw = self.corners_head(fused)

        return {
            "lambda_home_goals": torch.exp(goals_raw[:, 0]),
            "lambda_away_goals": torch.exp(goals_raw[:, 1]),
            "mean_home_shots": torch.exp(shots_raw[:, 0]),
            "mean_away_shots": torch.exp(shots_raw[:, 1]),
            "disp_home_shots": torch.exp(shots_raw[:, 2]),
            "disp_away_shots": torch.exp(shots_raw[:, 3]),
            "mean_home_corners": torch.exp(corners_raw[:, 0]),
            "mean_away_corners": torch.exp(corners_raw[:, 1]),
            "disp_home_corners": torch.exp(corners_raw[:, 2]),
            "disp_away_corners": torch.exp(corners_raw[:, 3]),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dl_model.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add dl_model/model.py tests/test_dl_model.py
git commit -m "Add CNN+Transformer multi-task model architecture"
```

---

### Task 4: Evaluation helpers (`dl_model/evaluation.py`)

**Files:**
- Create: `dl_model/evaluation.py`
- Test: `tests/test_dl_evaluation.py`

**Interfaces:**
- Consumes: `match_outcome_probs` (from `dixon_coles.py`), `implied_probabilities`, `log_loss`, `TARGET_LEAGUES` (from `evaluate_dixon_coles.py`), `MatchDataset` (from `dl_model.data`, Task 2), `negative_binomial_nll` (from `dl_model.losses`, Task 1).
- Produces: `poisson_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> numpy.ndarray` (no τ correction — this is the DL model's own score matrix, distinct from Dixon-Coles' `predict_score_matrix`), `evaluate_goals(model, matches: pandas.DataFrame, division_vocab, team_vocab, means, stds, batch_size: int = 512) -> pandas.DataFrame` (columns `MatchDate, model_log_loss, market_log_loss` — same shape as `evaluate_dixon_coles.walk_forward_evaluate`'s output, for direct side-by-side comparison), `evaluate_counts(model, matches: pandas.DataFrame, division_vocab, team_vocab, means, stds, batch_size: int = 512) -> pandas.DataFrame` (columns `shots_nll, corners_nll`, one row per match).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dl_evaluation.py
import numpy as np
import pandas as pd
import torch
from dl_model.data import CONTINUOUS_COLUMNS, build_vocab, compute_normalization_stats
from dl_model.model import MultiTaskMatchModel
from dl_model.evaluation import poisson_score_matrix, evaluate_goals, evaluate_counts


def test_poisson_score_matrix_sums_to_one_and_has_no_tau_artifact():
    matrix = poisson_score_matrix(1.5, 1.2, max_goals=10)
    assert matrix.shape == (11, 11)
    assert abs(matrix.sum() - 1.0) < 1e-6
    assert matrix[0, 0] > 0


def _sample_matches():
    data = {
        "Division": ["I1", "E0"],
        "MatchDate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "HomeTeam": ["Juventus", "Arsenal"],
        "AwayTeam": ["Milan", "Chelsea"],
        "FTHome": [2, 0],
        "FTAway": [1, 0],
        "FTResult": ["H", "D"],
        "OddHome": [1.8, 2.2],
        "OddDraw": [3.5, 3.1],
        "OddAway": [4.0, 3.4],
        "HomeShots": [15, 8],
        "AwayShots": [9, 7],
        "HomeCorners": [6, 3],
        "AwayCorners": [3, 2],
        "has_stats": [True, True],
    }
    for i, col in enumerate(CONTINUOUS_COLUMNS):
        data[col] = [0.1 + i * 0.01, -0.2 + i * 0.01]
    return pd.DataFrame(data)


def _sample_model_and_vocabs(matches):
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(matches)
    model = MultiTaskMatchModel(num_divisions=len(division_vocab), num_teams=len(team_vocab))
    return model, division_vocab, team_vocab, means, stds


def test_evaluate_goals_returns_expected_columns():
    matches = _sample_matches()
    model, division_vocab, team_vocab, means, stds = _sample_model_and_vocabs(matches)

    result = evaluate_goals(model, matches, division_vocab, team_vocab, means, stds)

    assert set(result.columns) == {"MatchDate", "model_log_loss", "market_log_loss"}
    assert len(result) == 2
    assert np.isfinite(result["model_log_loss"]).all()


def test_evaluate_counts_returns_expected_columns():
    matches = _sample_matches()
    model, division_vocab, team_vocab, means, stds = _sample_model_and_vocabs(matches)

    result = evaluate_counts(model, matches, division_vocab, team_vocab, means, stds)

    assert set(result.columns) == {"shots_nll", "corners_nll"}
    assert len(result) == 2
    assert np.isfinite(result["shots_nll"]).all()
    assert np.isfinite(result["corners_nll"]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dl_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dl_model.evaluation'`

- [ ] **Step 3: Write the minimal implementation**

```python
# dl_model/evaluation.py
import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson
from torch.utils.data import DataLoader

from dixon_coles import match_outcome_probs
from evaluate_dixon_coles import implied_probabilities, log_loss
from dl_model.data import MatchDataset
from dl_model.losses import negative_binomial_nll


def poisson_score_matrix(lambda_home, lambda_away, max_goals=10):
    goals = np.arange(max_goals + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    matrix = np.outer(p_home, p_away)
    return matrix / matrix.sum()


def evaluate_goals(model, matches, division_vocab, team_vocab, means, stds, batch_size=512):
    model.eval()
    match_rows = matches.reset_index(drop=True)
    dataset = MatchDataset(match_rows, division_vocab, team_vocab, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records = []
    row_offset = 0
    with torch.no_grad():
        for continuous, division_idx, home_idx, away_idx, _targets in loader:
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
                    "MatchDate": row["MatchDate"],
                    "model_log_loss": model_ll,
                    "market_log_loss": market_ll,
                })

            row_offset += batch_len

    return pd.DataFrame(records)


def evaluate_counts(model, matches, division_vocab, team_vocab, means, stds, batch_size=512):
    model.eval()
    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records = []
    with torch.no_grad():
        for continuous, division_idx, home_idx, away_idx, targets in loader:
            outputs = model(continuous, division_idx, home_idx, away_idx)

            shots_nll = negative_binomial_nll(
                outputs["mean_home_shots"], outputs["disp_home_shots"], targets["shots_home"]
            ) + negative_binomial_nll(
                outputs["mean_away_shots"], outputs["disp_away_shots"], targets["shots_away"]
            )
            corners_nll = negative_binomial_nll(
                outputs["mean_home_corners"], outputs["disp_home_corners"], targets["corners_home"]
            ) + negative_binomial_nll(
                outputs["mean_away_corners"], outputs["disp_away_corners"], targets["corners_away"]
            )

            for i in range(continuous.shape[0]):
                records.append({
                    "shots_nll": shots_nll[i].item(),
                    "corners_nll": corners_nll[i].item(),
                })

    return pd.DataFrame(records)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dl_evaluation.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add dl_model/evaluation.py tests/test_dl_evaluation.py
git commit -m "Add goal log-loss and count NLL evaluation helpers"
```

---

### Task 5: Training loop and walk-forward orchestration (`dl_model/train.py`)

**Files:**
- Create: `dl_model/train.py`
- Test: `tests/test_dl_train.py`

**Interfaces:**
- Consumes: `CONTINUOUS_COLUMNS`, `build_vocab`, `compute_normalization_stats`, `MatchDataset`, `load_all_matches` (from `dl_model.data`, Task 2); `MultiTaskMatchModel` (from `dl_model.model`, Task 3); `poisson_nll`, `negative_binomial_nll`, `masked_mean` (from `dl_model.losses`, Task 1); `evaluate_goals`, `evaluate_counts` (from `dl_model.evaluation`, Task 4); `TARGET_LEAGUES` (from `evaluate_dixon_coles.py`).
- Produces: `build_model_and_vocabs(matches) -> tuple[MultiTaskMatchModel, dict, dict, pandas.Series, pandas.Series]` (model, division_vocab, team_vocab, means, stds), `make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=256, shuffle=True) -> DataLoader`, `compute_total_loss(outputs: dict, targets: dict) -> Tensor` (scalar), `train_epochs(model, dataloader, optimizer, num_epochs) -> None` (mutates `model` in place), `run_walk_forward(matches, burn_in_seasons=5, initial_epochs=20, fine_tune_epochs=3) -> tuple[pandas.DataFrame, pandas.DataFrame]` (goal_results, count_results — same column shapes as `evaluate_goals`/`evaluate_counts`, concatenated across all evaluated seasons), plus a `__main__` block that loads the real dataset, runs the walk-forward, and prints a summary report.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dl_train.py
import numpy as np
import pandas as pd
import torch
from dl_model.data import CONTINUOUS_COLUMNS
from dl_model.train import (
    build_model_and_vocabs,
    make_dataloader,
    train_epochs,
    compute_total_loss,
    run_walk_forward,
)


def _fake_matches(n_seasons=6, teams=("A", "B", "C", "D"), divisions=("I1", "E0")):
    np.random.seed(7)
    rows = []
    for season_idx in range(n_seasons):
        season_start = pd.Timestamp("2015-08-01") + pd.DateOffset(years=season_idx)
        season_label = f"{season_start.year}-{season_start.year + 1}"
        for division in divisions:
            for round_num in range(8):
                date = season_start + pd.Timedelta(days=round_num * 7)
                shuffled = list(teams)
                np.random.shuffle(shuffled)
                for i in range(0, len(shuffled), 2):
                    h, a = shuffled[i], shuffled[i + 1]
                    gh, ga = int(np.random.poisson(1.4)), int(np.random.poisson(1.1))
                    sh, sa = int(np.random.poisson(12)), int(np.random.poisson(10))
                    ch, ca = int(np.random.poisson(5)), int(np.random.poisson(4))
                    result = "H" if gh > ga else ("A" if ga > gh else "D")
                    row = {
                        "Division": division, "MatchDate": date, "Season": season_label,
                        "HomeTeam": h, "AwayTeam": a, "FTHome": gh, "FTAway": ga, "FTResult": result,
                        "OddHome": 2.0, "OddDraw": 3.2, "OddAway": 3.8,
                        "HomeShots": sh, "AwayShots": sa, "HomeCorners": ch, "AwayCorners": ca,
                        "has_stats": True,
                    }
                    for col in CONTINUOUS_COLUMNS:
                        row[col] = float(np.random.randn())
                    rows.append(row)
    return pd.DataFrame(rows).sort_values("MatchDate").reset_index(drop=True)


def test_train_epochs_reduces_average_loss():
    matches = _fake_matches(n_seasons=2)
    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    loader = make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def average_loss():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for continuous, division_idx, home_idx, away_idx, targets in loader:
                outputs = model(continuous, division_idx, home_idx, away_idx)
                total += compute_total_loss(outputs, targets).item()
                count += 1
        return total / count

    loss_before = average_loss()
    train_epochs(model, loader, optimizer, num_epochs=10)
    loss_after = average_loss()

    assert loss_after < loss_before


def test_run_walk_forward_returns_expected_report_shapes():
    matches = _fake_matches(n_seasons=6)

    goal_results, count_results = run_walk_forward(
        matches, burn_in_seasons=3, initial_epochs=2, fine_tune_epochs=1
    )

    assert set(goal_results.columns) == {"MatchDate", "model_log_loss", "market_log_loss"}
    assert set(count_results.columns) == {"shots_nll", "corners_nll"}
    assert not goal_results.empty
    assert not count_results.empty
    assert np.isfinite(goal_results["model_log_loss"]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dl_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dl_model.train'`

- [ ] **Step 3: Write the minimal implementation**

```python
# dl_model/train.py
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_dixon_coles import TARGET_LEAGUES
from dl_model.data import build_vocab, compute_normalization_stats, MatchDataset, load_all_matches
from dl_model.model import MultiTaskMatchModel
from dl_model.losses import poisson_nll, negative_binomial_nll, masked_mean
from dl_model.evaluation import evaluate_goals, evaluate_counts

BURN_IN_SEASONS = 5
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.0005


def build_model_and_vocabs(matches):
    division_vocab = build_vocab(matches["Division"])
    team_vocab = build_vocab(pd.concat([matches["HomeTeam"], matches["AwayTeam"]]))
    means, stds = compute_normalization_stats(matches)
    model = MultiTaskMatchModel(num_divisions=len(division_vocab), num_teams=len(team_vocab))
    return model, division_vocab, team_vocab, means, stds


def make_dataloader(matches, division_vocab, team_vocab, means, stds, batch_size=BATCH_SIZE,
                     shuffle=True, drop_last=False):
    dataset = MatchDataset(matches, division_vocab, team_vocab, means, stds)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def compute_total_loss(outputs, targets):
    goals_loss = (
        poisson_nll(outputs["lambda_home_goals"], targets["goals_home"]).mean()
        + poisson_nll(outputs["lambda_away_goals"], targets["goals_away"]).mean()
    )

    shots_loss_elementwise = (
        negative_binomial_nll(outputs["mean_home_shots"], outputs["disp_home_shots"], targets["shots_home"])
        + negative_binomial_nll(outputs["mean_away_shots"], outputs["disp_away_shots"], targets["shots_away"])
    )
    shots_loss = masked_mean(shots_loss_elementwise, targets["has_stats"])

    corners_loss_elementwise = (
        negative_binomial_nll(outputs["mean_home_corners"], outputs["disp_home_corners"], targets["corners_home"])
        + negative_binomial_nll(outputs["mean_away_corners"], outputs["disp_away_corners"], targets["corners_away"])
    )
    corners_loss = masked_mean(corners_loss_elementwise, targets["has_stats"])

    return goals_loss + shots_loss + corners_loss


def train_epochs(model, dataloader, optimizer, num_epochs):
    model.train()
    for _epoch in range(num_epochs):
        for continuous, division_idx, home_idx, away_idx, targets in dataloader:
            if continuous.shape[0] < 2:
                continue  # BatchNorm1d cannot compute batch statistics from a single sample

            optimizer.zero_grad()
            outputs = model(continuous, division_idx, home_idx, away_idx)
            loss = compute_total_loss(outputs, targets)
            loss.backward()
            optimizer.step()


def run_walk_forward(matches, burn_in_seasons=BURN_IN_SEASONS, initial_epochs=INITIAL_EPOCHS,
                      fine_tune_epochs=FINE_TUNE_EPOCHS):
    matches = matches.sort_values("MatchDate").reset_index(drop=True)
    seasons_in_order = list(dict.fromkeys(matches["Season"]))
    burn_in_season_labels = seasons_in_order[:burn_in_seasons]
    eval_seasons = seasons_in_order[burn_in_seasons:]

    model, division_vocab, team_vocab, means, stds = build_model_and_vocabs(matches)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    burn_in_matches = matches[matches["Season"].isin(burn_in_season_labels)]
    burn_in_loader = make_dataloader(burn_in_matches, division_vocab, team_vocab, means, stds, drop_last=True)
    train_epochs(model, burn_in_loader, optimizer, initial_epochs)

    all_goal_results = []
    all_count_results = []

    for season in eval_seasons:
        season_matches = matches[matches["Season"] == season]

        goal_eval_matches = season_matches[season_matches["Division"].isin(TARGET_LEAGUES)]
        if not goal_eval_matches.empty:
            all_goal_results.append(
                evaluate_goals(model, goal_eval_matches, division_vocab, team_vocab, means, stds)
            )

        count_eval_matches = season_matches[season_matches["has_stats"]]
        if not count_eval_matches.empty:
            all_count_results.append(
                evaluate_counts(model, count_eval_matches, division_vocab, team_vocab, means, stds)
            )

        fine_tune_loader = make_dataloader(
            season_matches, division_vocab, team_vocab, means, stds, drop_last=True
        )
        train_epochs(model, fine_tune_loader, optimizer, fine_tune_epochs)

    goal_results = pd.concat(all_goal_results, ignore_index=True) if all_goal_results else pd.DataFrame(
        columns=["MatchDate", "model_log_loss", "market_log_loss"]
    )
    count_results = pd.concat(all_count_results, ignore_index=True) if all_count_results else pd.DataFrame(
        columns=["shots_nll", "corners_nll"]
    )
    return goal_results, count_results


if __name__ == "__main__":
    matches = load_all_matches()
    goal_results, count_results = run_walk_forward(matches)

    print("Goals task (Big 5, model vs market):")
    print(f"  model log-loss:  {goal_results['model_log_loss'].mean():.4f}")
    print(f"  market log-loss: {goal_results['market_log_loss'].mean():.4f}")

    print("\nShots/corners task (has_stats rows, Negative-Binomial NLL):")
    print(f"  shots NLL:   {count_results['shots_nll'].mean():.4f}")
    print(f"  corners NLL: {count_results['corners_nll'].mean():.4f}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dl_train.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (14 Dixon-Coles tests + 4 + 5 + 2 + 3 + 2 = 16 new deep learning tests = 30 total)

- [ ] **Step 6: Run the real training report (slow — expect a long run)**

Run: `python -m dl_model.train`
Expected: prints goals task log-loss (model vs market) and shots/corners NLL. This will take a while (burn-in training + per-season fine-tuning over ~20 seasons on the full dataset) — consider timing a single burn-in epoch on a subsample first, the way the Dixon-Coles evaluation was timed before committing to a full run.

- [ ] **Step 7: Commit**

```bash
git add dl_model/train.py tests/test_dl_train.py
git commit -m "Add training loop and walk-forward fine-tuning orchestration"
```
