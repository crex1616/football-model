"""Fetch data from the CollegeFootballData (CFBD) API and cache it in data/raw/.

Every API call in the project goes through this module. Each fetch is saved as a
Parquet file the first time, and later calls read that file instead of calling the
API again. The free tier allows 1,000 calls per month, so caching matters.

Files in data/raw/ hold the API response as returned: CFBD's camelCase column
names and original values. Renaming and cleaning happen later in clean.py.
"""

import json
import os
from collections.abc import Callable
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import cfbd
import pandas as pd
from dotenv import load_dotenv

from football_model import config


def get_api_client() -> cfbd.ApiClient:
    """Build an authenticated CFBD client using the key in .env.

    Reads CFBD_API_KEY from the environment (loaded from the repo's .env file).
    The key is never printed or logged.

    Returns:
        A cfbd.ApiClient ready to pass to cfbd.GamesApi, cfbd.BettingApi, etc.
    """
    load_dotenv(config.ROOT / ".env")
    api_key = os.getenv("CFBD_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise RuntimeError(
            "CFBD_API_KEY is not set. Copy .env.example to .env and paste your key into it."
        )
    configuration = cfbd.Configuration(access_token=api_key)
    return cfbd.ApiClient(configuration)


def _cached(path: Path, fetch_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return the DataFrame cached at `path`, fetching and saving it if missing.

    Args:
        path: Parquet file to read from or write to.
        fetch_fn: Zero-argument function that calls the API and returns a DataFrame.
            It runs only when `path` does not exist yet.

    Returns:
        The cached (or freshly fetched) DataFrame.
    """
    if path.exists():
        print(f"Loaded from cache: {path.name}")
        return pd.read_parquet(path)

    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Fetched from API (1 call) and saved: {path.name}")
    return df


def _to_records(api_objects: list) -> list[dict]:
    """Convert cfbd response objects to plain dicts with JSON-friendly values.

    Going through JSON turns enums (e.g. season type) into plain strings and dates
    into ISO strings, so the result can be saved to Parquet.
    """
    return [json.loads(json.dumps(obj.to_dict(), default=_json_default)) for obj in api_objects]


def _json_default(value: object) -> object:
    """Tell json.dumps how to write the non-JSON types cfbd returns."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    raise TypeError(f"Cannot convert {type(value).__name__} to JSON")


def flatten_lines(records: list[dict]) -> pd.DataFrame:
    """Reshape betting-line records to one row per game x provider.

    CFBD returns one record per game with a nested `lines` list (one entry per
    sportsbook). This expands that list into rows and keeps the game fields
    alongside. Values are not changed; this is only a reshape.

    Args:
        records: Dicts as returned by the CFBD /lines endpoint.

    Returns:
        DataFrame with game columns (id, season, homeTeam, ...) plus line columns
        (provider, spread, formattedSpread, ...). Games with no lines are dropped.
    """
    rows = []
    for record in records:
        game_fields = {key: value for key, value in record.items() if key != "lines"}
        for line in record.get("lines") or []:
            rows.append({**game_fields, **line})
    return pd.DataFrame(rows)


def flatten_records(records: list[dict]) -> pd.DataFrame:
    """Flatten nested records to one row per record, nested keys joined with dots.

    For example, an advanced-stats record {"team": "X", "offense": {"ppa": 0.2}}
    becomes columns `team` and `offense.ppa`. Values are not changed; this is only a
    reshape. An empty response becomes an empty DataFrame.

    Args:
        records: Dicts as returned by _to_records.

    Returns:
        DataFrame with one row per record.
    """
    return pd.json_normalize(records)


def load_games(season: int) -> pd.DataFrame:
    """Load every game involving an FBS team for one season (regular + postseason).

    Costs one API call the first time; afterward it reads data/raw/games_{season}.parquet.
    FBS-vs-FCS games are included because the FBS filter matches either team.

    Args:
        season: Season year, e.g. 2024.

    Returns:
        Raw games DataFrame, one row per game, CFBD column names.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            games = cfbd.GamesApi(client).get_games(
                year=season,
                season_type=cfbd.SeasonType.BOTH,
                classification=cfbd.DivisionClassification.FBS,
            )
        return pd.DataFrame(_to_records(games))

    return _cached(config.RAW_DIR / f"games_{season}.parquet", fetch)


def load_lines(season: int) -> pd.DataFrame:
    """Load betting lines for one season (regular + postseason), all providers.

    Costs one API call the first time; afterward it reads data/raw/lines_{season}.parquet.
    Lines are a benchmark only and are never used as a model input.

    Args:
        season: Season year, e.g. 2024.

    Returns:
        One row per game x provider (see flatten_lines), CFBD column names and
        CFBD's sign convention for `spread` (not yet converted to ours).
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            lines = cfbd.BettingApi(client).get_lines(year=season, season_type=cfbd.SeasonType.BOTH)
        return flatten_lines(_to_records(lines))

    return _cached(config.RAW_DIR / f"lines_{season}.parquet", fetch)


def load_advanced_game_stats(season: int) -> pd.DataFrame:
    """Load per-game advanced stats (EPA/PPA per play, success rate, ...) for one season.

    One row per team per game, with nested offense/defense stats flattened to columns
    like `offense.ppa` and `defense.successRate`. Garbage-time plays are excluded:
    scoring in a decided game says little about team strength.

    Costs one API call the first time; afterward it reads
    data/raw/advanced_game_stats_{season}.parquet.

    Args:
        season: Season year, e.g. 2024.

    Returns:
        Raw advanced game stats, CFBD column names.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            stats = cfbd.StatsApi(client).get_advanced_game_stats(
                year=season, season_type=cfbd.SeasonType.BOTH, exclude_garbage_time=True
            )
        return flatten_records(_to_records(stats))

    return _cached(config.RAW_DIR / f"advanced_game_stats_{season}.parquet", fetch)


def load_recruiting(season: int) -> pd.DataFrame:
    """Load team recruiting rankings (247Sports composite) for one recruiting class.

    Costs one API call the first time; afterward it reads
    data/raw/recruiting_{season}.parquet.

    Args:
        season: Recruiting class year, e.g. 2024 (players who enrolled for the 2024 season).

    Returns:
        One row per team: year, rank, team, points.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            rankings = cfbd.RecruitingApi(client).get_team_recruiting_rankings(year=season)
        return flatten_records(_to_records(rankings))

    return _cached(config.RAW_DIR / f"recruiting_{season}.parquet", fetch)


def load_talent(season: int) -> pd.DataFrame:
    """Load the team talent composite (roster-wide recruiting ratings) for one season.

    Costs one API call the first time; afterward it reads data/raw/talent_{season}.parquet.

    Args:
        season: Season year, e.g. 2024.

    Returns:
        One row per team: year, team, talent.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            talent = cfbd.TeamsApi(client).get_talent(year=season)
        return flatten_records(_to_records(talent))

    return _cached(config.RAW_DIR / f"talent_{season}.parquet", fetch)


def load_returning_production(season: int) -> pd.DataFrame:
    """Load returning production (share of last season's PPA and usage returning).

    Known before the season starts, so it is a valid preseason feature.

    Costs one API call the first time; afterward it reads
    data/raw/returning_production_{season}.parquet.

    Args:
        season: Season year, e.g. 2024 (production returning from 2023).

    Returns:
        One row per team, CFBD column names (percentPPA, usage, ...).
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            returning = cfbd.PlayersApi(client).get_returning_production(year=season)
        return flatten_records(_to_records(returning))

    return _cached(config.RAW_DIR / f"returning_production_{season}.parquet", fetch)


def load_venues() -> pd.DataFrame:
    """Load every venue with its location (one API call the first time).

    Returns:
        One row per venue: id, name, city, state, latitude/longitude, and more.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            venues = cfbd.VenuesApi(client).get_venues()
        return flatten_records(_to_records(venues))

    return _cached(config.RAW_DIR / "venues.parquet", fetch)


def load_teams() -> pd.DataFrame:
    """Load every team in every division with its home location (one API call the first time).

    All divisions are included so that FCS opponents, which are pooled as "FCS" in the
    games table, can still be located by their CFBD id.

    Returns:
        One row per team: id, school, classification, conference, location fields.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            teams = cfbd.TeamsApi(client).get_teams()
        return flatten_records(_to_records(teams))

    return _cached(config.RAW_DIR / "teams.parquet", fetch)
