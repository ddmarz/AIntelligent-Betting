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
