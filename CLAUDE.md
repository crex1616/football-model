# CLAUDE.md

Read this file before writing or changing any code in this repository. It defines how the project is organized and the rules every piece of code must follow, so that work done across many sessions stays consistent.

## About the project

A college football (FBS) prediction model: point spreads, win probabilities, season simulations, and team rankings. The full plan is in `docs/project-plan.md`; the build phases there are the roadmap. Check the Status table in `README.md` to see which phase is current, and update it when a phase is finished.

## About the user

The owner is learning machine learning. They have a strong background in regression and econometrics, but are new to ML practice and to production Python.

- **Explain as you go.** When introducing an ML concept (regularization, cross-validation, gradient boosting, calibration), connect it to its econometrics counterpart. Keep explanations short and put them in docstrings or comments where they help later.
- **Prefer clear over clever.** Readable pandas and plain functions over dense one-liners, metaprogramming, or heavy abstractions.
- **Build incrementally.** One phase at a time; don't scaffold future phases ahead of need.
- **Don't hide the statistics.** Write models so the user can see the design matrix, coefficients, and residuals.

## Tech stack

- Python 3.11+
- pandas for data, numpy for math
- statsmodels when econometric-style output is useful (coefficient tables, standard errors)
- scikit-learn for ridge/logistic models, pipelines, and metrics
- LightGBM for gradient boosting (Phase 5 only)
- matplotlib for plots
- cfbd (official CollegeFootballData.com client) for data
- pytest for tests

Add a dependency only when it's needed, and add it to `requirements.txt` in the same change.

## Repository layout

```
data/raw/          Untouched API responses saved as Parquet. Never edited by hand. Not committed.
data/interim/      Cleaned tables. Not committed.
data/processed/    Model-ready tables. Not committed.
docs/              Project plan and notes.
notebooks/         Exploration only. Numbered: 01_explore_games.ipynb, 02_elo.ipynb, ...
predictions/YYYY/  Weekly predictions, committed, never edited after kickoff.
reports/figures/   Saved plots.
scripts/           Thin command-line entry points that call code in src/.
src/football_model/ All reusable code.
tests/             pytest tests.
```

Planned modules in `src/football_model/` (create each when its phase starts):

| Module | Purpose | Phase |
| --- | --- | --- |
| `config.py` | Paths, season ranges, constants | 1 |
| `data.py` | Fetch from CFBD, cache to `data/raw/`, load cached files | 1 |
| `clean.py` | Build the canonical games table | 1 |
| `elo.py` | Elo ratings | 2 |
| `spread.py` | Ridge regression on point margin | 3 |
| `win_prob.py` | Spread to probability conversion | 4 |
| `features.py` | Pre-game feature construction | 5 |
| `evaluate.py` | Walk-forward backtests and metrics | 2+ |
| `simulate.py` | Monte Carlo season simulation | 6 |
| `rankings.py` | Power and résumé rankings | 7 |

**Notebooks are for exploring; `src/` is for code that lasts.** Once logic in a notebook works and will be reused, move it into a module and import it.

## Rules that must never be broken

### 1. No data leakage

Every feature for a game must be knowable before that game kicks off.

- Rolling or season-to-date stats exclude the game being predicted and every later game. Compute them with a shift (for example `groupby(...).shift(1)` before rolling) or by filtering on `start_date < game_date`.
- Ratings used to predict a game are the ratings as they stood *before* that game.
- Closing spreads and betting lines are a **benchmark only**, never a model input.
- Any new feature function gets a test confirming it doesn't use future games.

### 2. Validate in time order only

- Use walk-forward validation: train through season N, test on season N+1; within a season, predict week W from weeks before W.
- Never use random k-fold cross-validation or random train/test splits on game data.
- Tune hyperparameters (Elo K, ridge alpha) on validation seasons. Keep the most recent completed season as a final test set and evaluate on it rarely.

### 3. Protect the API quota and key

- The free CFBD tier allows 1,000 calls per month. Every API call goes through `data.py`, which saves the response to `data/raw/` and reads the cached file if it exists.
- Completed seasons are fetched once. In season, only the current week is re-fetched.
- The API key is read from the `CFBD_API_KEY` environment variable (loaded from `.env`). Never hard-code it, print it, or commit `.env`.

### 4. Predictions are permanent

- Weekly predictions are written to `predictions/YYYY/week_WW.csv` with a `created_at` timestamp before the first game of that week.
- Never overwrite or edit a prediction file once games have started. Corrections go in a new file with a note.

## Data conventions

- **Columns:** `snake_case`.
- **Game orientation:** one row per game from the home team's perspective. Columns use `home_` and `away_` prefixes (`home_team`, `away_points`).
- **Neutral sites:** keep CFBD's listed home team and set `neutral_site = True`; home-field advantage is zeroed for these games.
- **Margin:** `margin = home_points - away_points`. Positive means the home team won.
- **Spread sign:** a predicted or market spread is expressed as expected home margin. A home team favored by 7 is `+7.0`. (Sportsbooks print this as −7; convert on load and note it in the code.)
- **Team names:** use CFBD's `school` name as the identifier everywhere. Don't create alternate spellings.
- **FCS opponents:** map every FCS team to the single team `"FCS"` unless a phase explicitly changes this.
- **Keys:** `game_id` (CFBD id), `season`, `week`, `season_type` (`regular` / `postseason`).
- **Dates:** timezone-aware UTC timestamps.
- **File format:** Parquet for data in `data/`; CSV for `predictions/` so it's easy to read on GitHub.
- **2020 season:** flagged with `covid_season = True`; models must state whether they include it.

## Code style

- Type hints on all function signatures.
- A docstring on every public function: what it does, inputs, outputs, and for modeling code a line or two of the statistical idea.
- Functions take a DataFrame and return a new one; don't mutate inputs in place.
- No hidden global state. Settings come from `config.py` or function arguments.
- Set random seeds explicitly (`config.RANDOM_SEED`) for anything stochastic: simulations, boosting, sampling.
- Paths come from `config.py` (built with `pathlib`), never hard-coded strings.
- Format with `ruff format` and lint with `ruff check`.

## Evaluation conventions

- Spread models report mean absolute error in points.
- Probability models report log loss and Brier score, plus a calibration plot saved to `reports/figures/`.
- Every evaluation also scores the closing market line on the same games, so results are always shown as "model vs. market."
- A new feature or model is kept only if it improves walk-forward log loss (or MAE for spreads). Record the comparison in the commit message or `docs/`.

## Testing

- Tests live in `tests/`, named `test_<module>.py`, and run with `pytest`.
- Priorities: leakage checks for features, Elo math on a hand-calculated example, spread-sign conventions, and simulation probabilities summing correctly.
- Tests use small hand-built DataFrames, never live API calls.

## Git workflow

- Commit small, focused changes with messages that say what changed and why (for example `Add Elo margin-of-victory multiplier; log loss 0.582 -> 0.574`).
- Never commit `data/` contents (except `.gitkeep`), `.env`, or notebook outputs with large embedded data.
- Update the README Status table when a phase is complete.
