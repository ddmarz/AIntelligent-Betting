"""Calibration of the model's over/under and 1X2 probabilities.

Reads the per-match CSVs saved by a walk-forward run and measures how reliable the model's
probabilities are. See the "Calibrazione" section of
docs/superpowers/specs/2026-09-06-deep-learning-multitask-design.md.

This module must not import torch (directly or through other dl_model modules): in this
environment numpy/MKL and torch load two OpenMP runtimes and the process can abort.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

MAX_COUNT = 60


def _nb_params(mean, disp):
    mean = np.asarray(mean, dtype=float)
    disp = np.asarray(disp, dtype=float)
    # Same parametrization as dl_model.losses.negative_binomial_nll
    return disp, disp / (disp + mean)


def nb_pmf(mean, disp, max_count=MAX_COUNT):
    """Negative-binomial pmf over 0..max_count for each row: shape (n, max_count + 1)."""
    n, p = _nb_params(mean, disp)
    counts = np.arange(max_count + 1)
    return nbinom.pmf(counts[None, :], n[:, None], p[:, None])


def prob_over_single(mean, disp, line):
    """P(X > line) for a single team's count."""
    n, p = _nb_params(mean, disp)
    return nbinom.sf(math.floor(line), n, p)


def prob_over_total(mean_home, disp_home, mean_away, disp_away, line, max_count=MAX_COUNT):
    """P(X_home + X_away > line), convolving the two (independent) team distributions."""
    pmf_home = nb_pmf(mean_home, disp_home, max_count)
    pmf_away = nb_pmf(mean_away, disp_away, max_count)
    counts = np.arange(max_count + 1)
    at_most_line = (counts[:, None] + counts[None, :]) <= math.floor(line)
    # 1 - P(total <= line): mass truncated beyond max_count counts as "over"
    return 1.0 - np.einsum("ni,nj,ij->n", pmf_home, pmf_away, at_most_line)


def prob_over_goals(lambda_home, lambda_away, line):
    """P(total goals > line): the sum of two independent Poissons is Poisson(lambda_home + lambda_away)."""
    total = np.asarray(lambda_home, dtype=float) + np.asarray(lambda_away, dtype=float)
    return poisson.sf(math.floor(line), total)


def market_prob_over(over_odds, under_odds):
    """Market probability of the over with the bookmaker margin removed proportionally."""
    inv_over = 1.0 / np.asarray(over_odds, dtype=float)
    inv_under = 1.0 / np.asarray(under_odds, dtype=float)
    return inv_over / (inv_over + inv_under)


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion; (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half_width, center + half_width


def reliability_table(pred, outcome, n_bins=10):
    """Predicted probability in n_bins equal-width bins vs observed frequency, with Wilson 95% intervals."""
    pred = np.asarray(pred, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    # Clip both ends: P = 1.0 goes in the last bin, a rounding -1e-16 in the first one
    bin_index = np.clip(np.floor(pred * n_bins).astype(int), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        n = int(in_bin.sum())
        if n:
            mean_pred = float(pred[in_bin].mean())
            obs_freq = float(outcome[in_bin].mean())
            ci_low, ci_high = wilson_interval(float(outcome[in_bin].sum()), n)
        else:
            mean_pred = obs_freq = ci_low = ci_high = float("nan")
        rows.append({
            "bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "n": n,
            "mean_pred": mean_pred, "obs_freq": obs_freq, "ci_low": ci_low, "ci_high": ci_high,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(table):
    """Mean |predicted - observed| over the bins, weighted by the number of matches in each bin."""
    filled = table[table["n"] > 0]
    if filled.empty:
        return float("nan")
    weights = filled["n"] / filled["n"].sum()
    return float((weights * (filled["mean_pred"] - filled["obs_freq"]).abs()).sum())


def brier_skill_score(pred, outcome):
    """1 - Brier(model) / Brier(always predicting the base rate): 0 = adds nothing beyond the
    average frequency of the event, 1 = perfect. NaN when the event never or always happens."""
    pred = np.asarray(pred, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    base_rate = outcome.mean()
    base_brier = base_rate * (1 - base_rate)
    if base_brier == 0:
        return float("nan")
    return float(1 - np.mean((pred - outcome) ** 2) / base_brier)


def confidence_summary(pred, outcome, threshold=0.7):
    """How often the model is confident (P(over) >= threshold or <= 1 - threshold) and how often it is right.

    Confident unders are reported from the under's side: mean predicted P(under) and observed under frequency.
    """
    pred = np.asarray(pred, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    confident_over = pred >= threshold
    confident_under = pred <= 1 - threshold

    def side(mask, side_pred, side_outcome):
        n = int(mask.sum())
        if not n:
            return n, float("nan"), float("nan"), float("nan"), float("nan")
        ci_low, ci_high = wilson_interval(float(side_outcome[mask].sum()), n)
        return n, float(side_pred[mask].mean()), float(side_outcome[mask].mean()), ci_low, ci_high

    over = side(confident_over, pred, outcome)
    under = side(confident_under, 1 - pred, 1 - outcome)
    summary = {"share_confident": float((confident_over | confident_under).mean()) if len(pred) else float("nan")}
    for prefix, values in (("over", over), ("under", under)):
        for key, value in zip(("n", "mean_pred", "obs", "ci_low", "ci_high"), values):
            summary[f"{prefix}_{key}"] = value
    return summary


# ---------------------------------------------------------------------------
# Report from a run's per-match CSVs
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

CORNERS_TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
CORNERS_TEAM_LINES = [2.5, 3.5, 4.5, 5.5, 6.5]
SOT_TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
SOT_TEAM_LINES = [2.5, 3.5, 4.5, 5.5]
GOALS_LINES = [1.5, 2.5, 3.5]
MARKET_GOALS_LINE = 2.5
CONFIDENCE_THRESHOLD = 0.7

REQUIRED_GOALS_COLUMNS = [
    "FTResult", "FTHome", "FTAway", "Over25", "Under25", "lambda_home", "lambda_away",
    "model_p_home", "model_p_draw", "model_p_away", "market_p_home", "market_p_draw", "market_p_away",
]
REQUIRED_COUNTS_COLUMNS = [
    "HomeTarget", "AwayTarget", "HomeCorners", "AwayCorners",
    "pred_home_shots_on_target", "pred_away_shots_on_target", "pred_home_corners", "pred_away_corners",
    "disp_home_shots_on_target", "disp_away_shots_on_target", "disp_home_corners", "disp_away_corners",
]
GROUP_COLUMNS = ["family", "market", "line", "source"]

FAMILY_TITLES = {
    "corners_total": "Corner totali (over)",
    "corners_team": "Corner per squadra (over)",
    "sot_total": "Tiri in porta totali (over)",
    "sot_team": "Tiri in porta per squadra (over)",
    "goals_ou": "Gol totali (over)",
    "1x2": "1X2",
}

# family_total, family_team, lines_total, lines_team, home actual, away actual, model stat name
_COUNT_FAMILIES = [
    ("corners_total", "corners_team", CORNERS_TOTAL_LINES, CORNERS_TEAM_LINES, "HomeCorners", "AwayCorners", "corners"),
    ("sot_total", "sot_team", SOT_TOTAL_LINES, SOT_TEAM_LINES, "HomeTarget", "AwayTarget", "shots_on_target"),
]


def _events(family, market, line, source, pred, outcome):
    return pd.DataFrame({
        "family": family, "market": market, "line": line, "source": source,
        "pred": np.asarray(pred, dtype=float), "outcome": np.asarray(outcome, dtype=int),
    })


def build_events(goals, counts):
    """Long format, one row per (match, market, line, source): predicted P(event) and whether it happened."""
    frames = []

    for family_total, family_team, lines_total, lines_team, home_col, away_col, stat in _COUNT_FAMILIES:
        mean_home, disp_home = counts[f"pred_home_{stat}"].values, counts[f"disp_home_{stat}"].values
        mean_away, disp_away = counts[f"pred_away_{stat}"].values, counts[f"disp_away_{stat}"].values
        home, away = counts[home_col].values, counts[away_col].values
        for line in lines_total:
            pred = prob_over_total(mean_home, disp_home, mean_away, disp_away, line)
            frames.append(_events(family_total, f"over {line}", line, "model", pred, home + away > line))
        for line in lines_team:
            pred = np.concatenate([prob_over_single(mean_home, disp_home, line),
                                   prob_over_single(mean_away, disp_away, line)])
            outcome = np.concatenate([home > line, away > line])
            frames.append(_events(family_team, f"over {line}", line, "model", pred, outcome))

    total_goals = (goals["FTHome"] + goals["FTAway"]).values
    for line in GOALS_LINES:
        pred = prob_over_goals(goals["lambda_home"].values, goals["lambda_away"].values, line)
        frames.append(_events("goals_ou", f"over {line}", line, "model", pred, total_goals > line))
    # Missing O/U odds are stored as 0 in the dataset: keep only real quotes
    valid_odds = ((goals["Over25"] > 1) & (goals["Under25"] > 1)).values
    market_pred = market_prob_over(goals["Over25"].values[valid_odds], goals["Under25"].values[valid_odds])
    frames.append(_events("goals_ou", f"over {MARKET_GOALS_LINE}", MARKET_GOALS_LINE, "market",
                          market_pred, total_goals[valid_odds] > MARKET_GOALS_LINE))

    for label, result in (("home", "H"), ("draw", "D"), ("away", "A")):
        outcome = (goals["FTResult"] == result).values
        frames.append(_events("1x2", label, np.nan, "model", goals[f"model_p_{label}"].values, outcome))
        frames.append(_events("1x2", label, np.nan, "market", goals[f"market_p_{label}"].values, outcome))

    return pd.concat(frames, ignore_index=True)


def calibration_report(events, n_bins=10, threshold=CONFIDENCE_THRESHOLD):
    """Returns (summary, bins): one summary row per family/market/line/source, and every reliability bin."""
    summary_rows, bin_tables = [], []
    for keys, group in events.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        group_values = dict(zip(GROUP_COLUMNS, keys))
        pred, outcome = group["pred"].values, group["outcome"].values
        table = reliability_table(pred, outcome, n_bins)
        summary_rows.append({
            **group_values,
            "n": len(group),
            "base_rate": float(outcome.mean()),
            "brier_skill": brier_skill_score(pred, outcome),
            "ece": expected_calibration_error(table),
            **confidence_summary(group["pred"].values, group["outcome"].values, threshold),
        })
        bin_tables.append(table.assign(**group_values)[GROUP_COLUMNS + list(table.columns)])
    return pd.DataFrame(summary_rows), pd.concat(bin_tables, ignore_index=True)


def plot_reliability(bins, out_dir):
    """One reliability chart per family: perfect-calibration diagonal, one curve per market line
    (market reference dashed), point size proportional to the number of matches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    paths = []
    for family, family_bins in bins.groupby("family", sort=False):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], color="grey", linewidth=1, label="calibrazione perfetta")
        max_n = max(family_bins["n"].max(), 1)
        for (market, source), curve in family_bins.groupby(["market", "source"], sort=False):
            curve = curve[curve["n"] > 0]
            if curve.empty:
                continue
            label = f"{market} (mercato)" if source == "market" else market
            plotted, = ax.plot(curve["mean_pred"], curve["obs_freq"],
                               linestyle="--" if source == "market" else "-", label=label)
            ax.scatter(curve["mean_pred"], curve["obs_freq"], s=15 + 200 * curve["n"] / max_n,
                       color=plotted.get_color())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Probabilità predetta")
        ax.set_ylabel("Frequenza reale")
        ax.set_title(FAMILY_TITLES.get(family, family))
        ax.legend(fontsize=8, loc="upper left")
        path = out_dir / f"reliability_{family}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def find_run(checkpoint_dir=CHECKPOINT_DIR, run_name=None):
    """Run name (multitask_<timestamp>); the most recent one with per-match results when not given."""
    if run_name:
        return run_name
    candidates = sorted(Path(checkpoint_dir).glob("multitask_*_goals.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun run con risultati per partita in {checkpoint_dir}: lancia prima il training."
        )
    return candidates[-1].name[: -len("_goals.csv")]


def load_run(checkpoint_dir, run_name):
    checkpoint_dir = Path(checkpoint_dir)
    goals = pd.read_csv(checkpoint_dir / f"{run_name}_goals.csv")
    counts = pd.read_csv(checkpoint_dir / f"{run_name}_counts.csv")
    missing = [c for c in REQUIRED_GOALS_COLUMNS if c not in goals.columns]
    missing += [c for c in REQUIRED_COUNTS_COLUMNS if c not in counts.columns]
    if missing:
        raise ValueError(
            f"Il run {run_name} non ha le colonne {missing}: è stato fatto prima che la calibrazione "
            "fosse disponibile, rilancia il training."
        )
    return goals, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrazione delle probabilità di un run walk-forward.")
    parser.add_argument("run_name", nargs="?", help="nome del run (multitask_<data-ora>); default: il più recente")
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="soglia di 'convinto': P(over) >= soglia o <= 1 - soglia")
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    run_name = find_run(checkpoint_dir, args.run_name)
    goals, counts = load_run(checkpoint_dir, run_name)
    summary, bins = calibration_report(build_events(goals, counts), threshold=args.threshold)

    out_dir = checkpoint_dir / f"{run_name}_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    bins.to_csv(out_dir / "bins.csv", index=False)
    plot_paths = plot_reliability(bins, out_dir)

    columns = ["family", "market", "source", "n", "base_rate", "brier_skill", "ece", "share_confident",
               "over_n", "over_mean_pred", "over_obs", "under_n", "under_mean_pred", "under_obs"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(f"Calibrazione del run {run_name} (soglia 'convinto' {args.threshold:.0%})\n")
        print(summary[columns].round(3).to_string(index=False))
    print(f"\nTabelle e {len(plot_paths)} grafici salvati in {out_dir}")


if __name__ == "__main__":
    main()
