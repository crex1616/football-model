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
FIGURES_DIR = ROOT / "reports" / "figures"

# Seasons used for model history. 2020 is kept but flagged (COVID-disrupted).
FIRST_SEASON = 2014
CURRENT_SEASON = 2026
COVID_SEASON = 2020

# Seed for anything random (simulations, boosting, sampling) so results are reproducible.
RANDOM_SEED = 42
