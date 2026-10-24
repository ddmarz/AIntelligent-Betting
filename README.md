# AIntelligent-Betting

A side project to build a football match predictor and, eventually, a value-betting assistant. It is also a testbed for Claude Code skills I might reuse in more serious, private projects. Work and thesis come first, so updates will arrive when I have free time, hopefully after graduation.

## Goal

Predict match results and statistics (goals, shots, corners, possibly cards and fouls) with **well-calibrated confidence**. Then compare that confidence with pre-match odds to spot value bets.

Longer-term ideas:
- Suggest plays based on the user's target multiplier and number of games.
- Manage a virtual portfolio automatically and track how it performs.

Clearly, these ideas are ambitious and will require some time and work. Slow and steady wins the race I guess.

## Status

| Area | State |
|---|---|
| Dataset building and cleaning (`dataset.py`) | Done (may still change) |
| Deep-learning multi-task model (`dl_model/`: goals, shots, corners) | Work in progress |
| XGBoost baseline | Planned |
| Odds comparison / value detection | Planned |
| Virtual portfolio | Idea |

## Project layout

- `dataset.py`: builds the clean dataset (`matches_clean.csv`) from the raw match and Elo data, with team-name alias handling.
- `dl_model/`: data loading, model, losses, training and evaluation for the multi-task network.
- `tests/`: unit tests for the DL modules.
- `docs/`: design specs.
- `prove.ipynb`: scratch notebook for experiments.

## References

- XGabora. *Club-Football-Match-Data-2000-2025* (`Matches.csv`, `EloRatings.csv`). https://github.com/xgabora/Club-Football-Match-Data-2000-2025
- ClubElo. *Club Elo Ratings* (source of the Elo history). http://clubelo.com

Many thanks to both for the data, keep up the great work!
