"""Fase 1: pulizia dataset & feature engineering.

Vedi docs/superpowers/specs/2026-07-12-dataset-feature-engineering-design.md
per il design completo.
"""

import pandas as pd

# .csv file paths
RAW_MATCHES_PATH = "Matches.csv"
RAW_ELO_PATH = "EloRatings.csv"
TEAM_ALIASES_PATH = "team_aliases.csv"
OUTPUT_PATH = "matches_clean.csv"

# Constants
ROLLING_WINDOW = 5
SEASON_START_MONTH = 7

DROPPED_COLUMNS = [
    "MatchTime",
    "MaxHome", "MaxDraw", "MaxAway",
    "MaxOver25", "MaxUnder25",
]

# Feature groups for availability flags
STATS_COLUMNS = [
    "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
    "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners",
    "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
    "HTHome", "HTAway", "HTResult",
]
HANDICAP_COLUMNS = ["HandiSize", "HandiHome", "HandiAway"]
OU_COLUMNS = ["Over25", "Under25"]
CLUSTER_COLUMNS = ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]
FORM_COLUMNS = ["Form3Home", "Form5Home", "Form3Away", "Form5Away"]

# Availability flags indicate whether a match has data for a given feature group
FLAG_GROUPS = {
    "has_stats": STATS_COLUMNS,
    "has_handicap": HANDICAP_COLUMNS,
    "has_ou": OU_COLUMNS,
    "has_cluster_data": CLUSTER_COLUMNS,
    "has_form": FORM_COLUMNS,
}

METADATA_COLUMNS = ["Division", "MatchDate", "Season", "HomeTeam", "AwayTeam"]
TARGET_COLUMNS = ["FTHome", "FTAway", "FTResult"]


def load_matches(path=RAW_MATCHES_PATH):
    matches = pd.read_csv(path, low_memory=False, dtype={"MatchTime": str})
    matches["MatchDate"] = pd.to_datetime(matches["MatchDate"])
    return matches


def load_elo(path=RAW_ELO_PATH):
    elo = pd.read_csv(path)
    elo["date"] = pd.to_datetime(elo["date"])
    return elo


def load_team_aliases(path=TEAM_ALIASES_PATH):
    aliases = pd.read_csv(path)
    return dict(zip(aliases["MatchesTeam"], aliases["EloTeam"]))


def fill_missing_elo(matches, elo, team_aliases=None):
    team_aliases = team_aliases or {}
    matches = matches.sort_values("MatchDate")
    matches["HomeTeamForElo"] = matches["HomeTeam"].replace(team_aliases)
    matches["AwayTeamForElo"] = matches["AwayTeam"].replace(team_aliases)

    home_lookup = elo[["club", "date", "elo"]].rename(
        columns={"club": "HomeTeamForElo", "elo": "HomeEloLookup"}
    ).sort_values("date")
    away_lookup = elo[["club", "date", "elo"]].rename(
        columns={"club": "AwayTeamForElo", "elo": "AwayEloLookup"}
    ).sort_values("date")

    matches = pd.merge_asof(
        matches, home_lookup,
        left_on="MatchDate", right_on="date",
        by="HomeTeamForElo", direction="backward",
    ).drop(columns="date")

    matches = pd.merge_asof(
        matches, away_lookup,
        left_on="MatchDate", right_on="date",
        by="AwayTeamForElo", direction="backward",
    ).drop(columns="date")

    matches["HomeElo"] = matches["HomeElo"].fillna(matches["HomeEloLookup"])
    matches["AwayElo"] = matches["AwayElo"].fillna(matches["AwayEloLookup"])
    return matches.drop(columns=["HomeTeamForElo", "AwayTeamForElo", "HomeEloLookup", "AwayEloLookup"])


def drop_missing_odds(matches):
    return matches.dropna(subset=["OddHome", "OddDraw", "OddAway"])


def drop_missing_target(matches):
    return matches.dropna(subset=["FTHome", "FTAway", "FTResult"])


def drop_low_value_columns(matches):
    return matches.drop(columns=DROPPED_COLUMNS, errors="ignore")


def add_availability_flags(matches):
    for flag, cols in FLAG_GROUPS.items():
        present_cols = [c for c in cols if c in matches.columns]
        matches[flag] = matches[present_cols].notna().any(axis=1)
        for col in present_cols:
            if pd.api.types.is_numeric_dtype(matches[col]):
                matches[col] = matches[col].fillna(0)
            else:
                matches[col] = matches[col].fillna("Unknown")
    return matches


def add_elo_diff(matches):
    matches["has_elo"] = matches["HomeElo"].notna() & matches["AwayElo"].notna()
    matches["HomeElo"] = matches["HomeElo"].fillna(0)
    matches["AwayElo"] = matches["AwayElo"].fillna(0)
    matches["EloDiff"] = matches["HomeElo"] - matches["AwayElo"]
    return matches


def add_rolling_goal_features(matches, window=ROLLING_WINDOW):
    matches = matches.reset_index(drop=True)
    matches["match_id"] = matches.index

    home_rows = matches[["match_id", "MatchDate", "HomeTeam", "FTHome", "FTAway"]].rename(
        columns={"HomeTeam": "Team", "FTHome": "GoalsFor", "FTAway": "GoalsAgainst"}
    )
    home_rows["Side"] = "Home"

    away_rows = matches[["match_id", "MatchDate", "AwayTeam", "FTAway", "FTHome"]].rename(
        columns={"AwayTeam": "Team", "FTAway": "GoalsFor", "FTHome": "GoalsAgainst"}
    )
    away_rows["Side"] = "Away"

    history = pd.concat([home_rows, away_rows], ignore_index=True)
    history = history.sort_values(["Team", "MatchDate", "match_id"])

    grouped = history.groupby("Team")
    history["RollingGoalsFor"] = grouped["GoalsFor"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    history["RollingGoalsAgainst"] = grouped["GoalsAgainst"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    history["PriorMatchCount"] = grouped.cumcount()

    home_feats = history[history["Side"] == "Home"][
        ["match_id", "RollingGoalsFor", "RollingGoalsAgainst", "PriorMatchCount"]
    ].rename(columns={
        "RollingGoalsFor": "HomeAvgGoalsFor5",
        "RollingGoalsAgainst": "HomeAvgGoalsAgainst5",
        "PriorMatchCount": "HomePriorMatchCount",
    })

    away_feats = history[history["Side"] == "Away"][
        ["match_id", "RollingGoalsFor", "RollingGoalsAgainst", "PriorMatchCount"]
    ].rename(columns={
        "RollingGoalsFor": "AwayAvgGoalsFor5",
        "RollingGoalsAgainst": "AwayAvgGoalsAgainst5",
        "PriorMatchCount": "AwayPriorMatchCount",
    })

    matches = matches.merge(home_feats, on="match_id").merge(away_feats, on="match_id")

    matches["has_rolling_stats"] = (
        (matches["HomePriorMatchCount"] >= window) & (matches["AwayPriorMatchCount"] >= window)
    )

    rolling_cols = [
        "HomeAvgGoalsFor5", "HomeAvgGoalsAgainst5",
        "AwayAvgGoalsFor5", "AwayAvgGoalsAgainst5",
    ]
    matches[rolling_cols] = matches[rolling_cols].fillna(0)

    return matches.drop(columns=["match_id", "HomePriorMatchCount", "AwayPriorMatchCount"])


def add_season(matches):
    def season_label(date):
        year = date.year
        if date.month >= SEASON_START_MONTH:
            return f"{year}-{year + 1}"
        return f"{year - 1}-{year}"

    matches["Season"] = matches["MatchDate"].apply(season_label)
    return matches


def select_output_columns(matches):
    flag_columns = list(FLAG_GROUPS.keys()) + ["has_elo", "has_rolling_stats"]
    feature_columns = [
        c for c in matches.columns
        if c not in METADATA_COLUMNS + TARGET_COLUMNS + flag_columns
    ]
    ordered = METADATA_COLUMNS + TARGET_COLUMNS + feature_columns + flag_columns
    return matches[ordered]


def build_clean_dataset(matches_path=RAW_MATCHES_PATH, elo_path=RAW_ELO_PATH, aliases_path=TEAM_ALIASES_PATH):
    matches = load_matches(matches_path)
    elo = load_elo(elo_path)
    team_aliases = load_team_aliases(aliases_path)

    matches = fill_missing_elo(matches, elo, team_aliases)
    matches = drop_missing_target(matches)
    matches = drop_missing_odds(matches)
    matches = drop_low_value_columns(matches)
    matches = add_availability_flags(matches)
    matches = add_elo_diff(matches)
    matches = add_rolling_goal_features(matches)
    matches = add_season(matches)
    matches = matches.sort_values("MatchDate").reset_index(drop=True)
    return select_output_columns(matches)


if __name__ == "__main__":
    clean = build_clean_dataset()
    clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(clean)} rows, {len(clean.columns)} columns to {OUTPUT_PATH}")
