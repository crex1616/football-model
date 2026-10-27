"""Build the canonical games table from raw CFBD games and betting lines.

The output has one row per completed game from the home team's perspective, in
snake_case, sorted by kickoff time. Decisions behind each step are documented in
notebooks/01_explore_games.ipynb (section 9).

Post-game columns (quarter scores, postgame win probability and Elo, excitement,
attendance) are dropped here so they can never leak into a model.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from football_model import config, data

# Raw CFBD column -> our column. Anything not listed is dropped, which is how the
# post-game (leaky) columns are removed.
GAME_COLUMNS = {
    "id": "game_id",
    "season": "season",
    "week": "week",
    "seasonType": "season_type",
    "startDate": "start_date",
    "neutralSite": "neutral_site",
    "conferenceGame": "conference_game",
    "venue": "venue",
    # CFBD ids. home_id/away_id keep the real school even after non-FBS teams are
    # pooled as "FCS", so their locations can still be looked up (travel distance).
    "venueId": "venue_id",
    "homeId": "home_id",
    "awayId": "away_id",
    "homeTeam": "home_team",
    "homeConference": "home_conference",
    "homeClassification": "home_classification",
    "homePoints": "home_points",
    "awayTeam": "away_team",
    "awayConference": "away_conference",
    "awayClassification": "away_classification",
    "awayPoints": "away_points",
    # CFBD's own pregame Elo: known before kickoff, but it's someone else's model.
    # Kept for comparison only; never use it as a feature.
    "homePregameElo": "cfbd_home_elo",
    "awayPregameElo": "cfbd_away_elo",
    "notes": "notes",
}

FCS_TEAM = "FCS"


def _select_columns(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Keep and rename the pre-game-safe columns, parse kickoffs to UTC, pool FCS teams.

    The steps shared by clean_games (finished games) and clean_schedule (unplayed ones).
    """
    games = raw_games[list(GAME_COLUMNS)].rename(columns=GAME_COLUMNS)
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True)

    # One shared "FCS" team: individual FCS teams play only 1-2 FBS games a season,
    # far too few to estimate a rating each. Anything not labeled "fbs" (FCS, D-II,
    # missing) goes here. The original classification columns are kept for reference.
    for side in ["home", "away"]:
        is_fbs = games[f"{side}_classification"] == "fbs"
        games[f"{side}_team"] = games[f"{side}_team"].where(is_fbs, FCS_TEAM)
    return games


def _is_completed(raw_games: pd.DataFrame) -> pd.Series:
    """True for games marked completed with both scores present."""
    completed = raw_games["completed"].fillna(False).astype(bool)
    return completed & raw_games["homePoints"].notna() & raw_games["awayPoints"].notna()


def clean_games(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Turn raw CFBD games into the clean, home-perspective games table.

    Steps: drop unfinished games, keep and rename the pre-game-safe columns, parse
    kickoff times to UTC, map every non-FBS team to the single team "FCS", and add
    the target columns `margin` and `home_win`.

    Args:
        raw_games: Games as returned by data.load_games (CFBD camelCase columns).
            May hold several seasons stacked together.

    Returns:
        New DataFrame, one row per completed game, sorted by start_date.
    """
    completed = _is_completed(raw_games)
    n_dropped = int((~completed).sum())
    if n_dropped:
        print(f"Dropped {n_dropped} game(s) that were not completed or have no score.")

    games = _select_columns(raw_games.loc[completed])
    games["home_points"] = games["home_points"].astype(int)
    games["away_points"] = games["away_points"].astype(int)
    games["margin"] = games["home_points"] - games["away_points"]
    games["home_win"] = (games["margin"] > 0).astype(int)
    games["covid_season"] = games["season"] == config.COVID_SEASON

    # Sort by kickoff, not week: every postseason game is labeled week 1.
    return games.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def clean_schedule(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Unplayed games in the same shape as clean_games, without scores or targets.

    These are the games to predict. Points, margin, and home_win are dropped because
    they don't exist yet.

    Args:
        raw_games: Games as returned by data.load_games_week (CFBD camelCase columns).

    Returns:
        New DataFrame, one row per game not yet completed, sorted by start_date.
    """
    games = _select_columns(raw_games.loc[~_is_completed(raw_games)])
    games = games.drop(columns=["home_points", "away_points"])
    games["covid_season"] = games["season"] == config.COVID_SEASON
    return games.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def build_current_season(season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Games played so far this regular season, and the games of the week to predict.

    Loads regular-season weeks 1 through `week` with data.load_games_week (cached; only
    weeks still in progress are re-fetched) and the week's lines with
    data.load_lines_week. Nothing is written to games.parquet, which holds completed
    seasons only. Postseason weeks are not handled yet.

    Args:
        season: The season in progress.
        week: The regular-season week to predict.

    Returns:
        (completed, upcoming): completed is clean_games output for every finished game
        this season; upcoming is clean_schedule output for the unplayed games of `week`,
        plus market_spread and n_providers (the line as of now; benchmark only).
    """
    raw = pd.concat(
        [data.load_games_week(season, w) for w in range(1, week + 1)], ignore_index=True
    )
    upcoming = clean_schedule(raw[raw["week"] == week])

    lines = data.load_lines_week(season, week)
    if lines.empty:
        upcoming["market_spread"] = np.nan
        upcoming["n_providers"] = 0
    else:
        upcoming = add_market_lines(upcoming, consensus_lines(lines))

    completed = clean_games(raw[_is_completed(raw)])
    return completed, upcoming


def consensus_lines(raw_lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse betting lines to one closing spread per game, in our sign convention.

    CFBD's `spread` is the latest (closing) line in sportsbook style: negative when
    the home team is favored. Our convention is expected home margin, so a home
    favorite by 7 becomes +7.0. The consensus is the mean across available books.

    Market lines are a benchmark only, never a model input.

    Args:
        raw_lines: Lines as returned by data.load_lines (one row per game x provider).

    Returns:
        New DataFrame with columns game_id, market_spread, n_providers.
    """
    lines = raw_lines.dropna(subset=["spread"])
    consensus = (
        lines.groupby("id")
        .agg(cfbd_spread=("spread", "mean"), n_providers=("provider", "nunique"))
        .reset_index()
        .rename(columns={"id": "game_id"})
    )
    # Sportsbook spread is negative when home is favored; ours is expected home margin.
    consensus["market_spread"] = -consensus["cfbd_spread"]
    return consensus[["game_id", "market_spread", "n_providers"]]


def add_market_lines(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Left-join consensus market spreads onto the games table.

    Games without any posted line get NaN market_spread and n_providers = 0. When
    comparing model and market, score both on the games where market_spread exists.

    Args:
        games: Output of clean_games.
        lines: Output of consensus_lines.

    Returns:
        New DataFrame: games plus market_spread and n_providers.
    """
    merged = games.merge(lines, on="game_id", how="left", validate="one_to_one")
    merged["n_providers"] = merged["n_providers"].fillna(0).astype(int)
    return merged


def build_games_table(seasons: Iterable[int]) -> pd.DataFrame:
    """Load, clean, and join games and lines for several seasons, then save to disk.

    Uses the cached raw files in data/raw/ (fetching any missing season, 2 API calls
    per season) and writes config.GAMES_TABLE_PATH.

    Args:
        seasons: Season years to include, e.g. range(2014, 2026).

    Returns:
        The combined games table.
    """
    seasons = list(seasons)
    raw_games = pd.concat([data.load_games(s) for s in seasons], ignore_index=True)
    raw_lines = pd.concat([data.load_lines(s) for s in seasons], ignore_index=True)

    games = add_market_lines(clean_games(raw_games), consensus_lines(raw_lines))

    config.GAMES_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(config.GAMES_TABLE_PATH, index=False)
    print(f"Saved {len(games):,} games to {config.GAMES_TABLE_PATH}")
    return games


def load_games_table() -> pd.DataFrame:
    """Read the saved games table (run scripts/build_games.py first)."""
    return pd.read_parquet(config.GAMES_TABLE_PATH)


def market_mae(games: pd.DataFrame) -> float:
    """Mean absolute error of the market spread vs. actual margin, on games with a line."""
    has_line = games["market_spread"].notna()
    return float(
        np.mean(np.abs(games.loc[has_line, "margin"] - games.loc[has_line, "market_spread"]))
    )
