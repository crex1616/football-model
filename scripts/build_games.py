"""Build data/processed/games.parquet for every completed season.

Usage:
    python scripts/build_games.py

The first run fetches any season not yet in data/raw/ (2 API calls per season).
Later runs read the cache and make no API calls.
"""

from football_model import clean, config


def main() -> None:
    """Build and save the games table for FIRST_SEASON through the last completed season."""
    seasons = range(config.FIRST_SEASON, config.CURRENT_SEASON)
    games = clean.build_games_table(seasons)
    print(f"Seasons {seasons.start}-{seasons.stop - 1}: {len(games):,} games")
    print(f"Closing-line MAE: {clean.market_mae(games):.2f} points")


if __name__ == "__main__":
    main()
