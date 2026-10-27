"""Project-wide paths and constants.

Every other module imports paths and settings from here, so nothing is hard-coded twice.
"""

from pathlib import Path

# Repository root: src/football_model/config.py -> two levels up from src/
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = ROOT / "predictions"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# The canonical games table built by clean.py: one row per completed game, home perspective.
GAMES_TABLE_PATH = PROCESSED_DIR / "games.parquet"

# Pre-game features for every game (features.py), built by scripts/evaluate_features.py.
FEATURES_TABLE_PATH = PROCESSED_DIR / "features.parquet"

# Seasons used for model history. 2020 is kept but flagged (COVID-disrupted).
FIRST_SEASON = 2014
CURRENT_SEASON = 2026
COVID_SEASON = 2020

# Time zone for kickoff times printed to the terminal. Stored data stays in UTC.
DISPLAY_TIMEZONE = "America/Denver"

# Seed for anything random (simulations, boosting, sampling) so results are reproducible.
RANDOM_SEED = 42

# Validation splits (walk-forward). 2014 is a burn-in season for ratings that start
# from scratch, so it is never scored. Hyperparameters are tuned on TUNING_SEASONS;
# TEST_SEASON (the most recent completed season) is scored rarely, as a final check.
BURN_IN_SEASON = FIRST_SEASON
TEST_SEASON = CURRENT_SEASON - 1
TUNING_SEASONS = [s for s in range(BURN_IN_SEASON + 1, TEST_SEASON) if s != COVID_SEASON]
