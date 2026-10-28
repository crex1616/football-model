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


def _cached(
    path: Path,
    fetch_fn: Callable[[], pd.DataFrame],
    is_stale: Callable[[pd.DataFrame], bool] | None = None,
) -> pd.DataFrame:
    """Return the DataFrame cached at `path`, fetching and saving it if missing.

    Args:
        path: Parquet file to read from or write to.
        fetch_fn: Zero-argument function that calls the API and returns a DataFrame.
            It runs only when `path` does not exist yet (or the cached copy is stale).
        is_stale: Optional check on the cached DataFrame. If it returns True, the file
            is fetched again and overwritten. Used for in-season weeks, which change
            until every game is final. None means a cached file is never re-fetched.

    Returns:
        The cached (or freshly fetched) DataFrame.
    """
    if path.exists():
        cached = pd.read_parquet(path)
        if is_stale is None or not is_stale(cached):
            print(f"Loaded from cache: {path.name}")
            return cached
        print(f"Cached file is stale (games not final), re-fetching: {path.name}")

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


def flatten_polls(records: list[dict]) -> pd.DataFrame:
    """Reshape poll records to one row per season x week x poll x ranked team.

    CFBD nests ranks inside polls inside poll weeks. Values are not changed (CFBD's
    school names, rank 1 = best); this is only a reshape.

    Args:
        records: Dicts as returned by the CFBD /rankings endpoint (_to_records).

    Returns:
        Columns season, seasonType, week, poll, isFinal, rank, teamId, school,
        conference, firstPlaceVotes, points. Empty if there are no polls.
    """
    rows = []
    for week in records:
        for poll in week.get("polls") or []:
            for rank in poll.get("ranks") or []:
                rows.append(
                    {
                        "season": week.get("season"),
                        "seasonType": week.get("seasonType"),
                        "week": week.get("week"),
                        "poll": poll.get("poll"),
                        "isFinal": poll.get("isFinal"),
                        **rank,
                    }
                )
    return pd.DataFrame(rows)


def load_rankings(season: int) -> pd.DataFrame:
    """Load every poll (AP, Coaches, CFP committee, ...) for every week of a season.

    One API call the first time; afterward it reads data/raw/rankings_{season}.parquet.
    Call it only for completed seasons (in season, use load_rankings_week). Polls are a
    benchmark for the rankings, never a model input.

    Returns:
        One row per week x poll x ranked team (see flatten_polls).
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            weeks = cfbd.RankingsApi(client).get_rankings(year=season)
        return flatten_polls(_to_records(weeks))

    return _cached(config.RAW_DIR / f"rankings_{season}.parquet", fetch)


AP_POLL = "AP Top 25"
COMMITTEE_POLL = "Playoff Committee Rankings"
# CFBD labels a poll by the week it precedes. The committee's first ranking has been
# poll week 10 or 11 (2014-2025); from week 10 on, a week without it is re-fetched.
FIRST_COMMITTEE_POLL_WEEK = 10


def load_rankings_week(season: int, week: int) -> pd.DataFrame:
    """Load one regular-season poll week (AP, committee, ...) for the season in progress.

    Poll week W is released before week W's games, so the rankings after week 8 are poll
    week 9. The file is re-fetched while the AP poll is missing, or, from
    FIRST_COMMITTEE_POLL_WEEK on, while the committee ranking is missing (it comes out
    on Tuesdays, two days after the AP). About one call per week. A benchmark only.

    Returns:
        One row per poll x ranked team (see flatten_polls).
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            weeks = cfbd.RankingsApi(client).get_rankings(
                year=season, week=week, season_type=cfbd.SeasonType("regular")
            )
        return flatten_polls(_to_records(weeks))

    def missing_polls(polls: pd.DataFrame) -> bool:
        if polls.empty or AP_POLL not in set(polls["poll"]):
            return True
        return week >= FIRST_COMMITTEE_POLL_WEEK and COMMITTEE_POLL not in set(polls["poll"])

    path = config.RAW_DIR / f"rankings_{season}_regular_week_{week:02d}.parquet"
    return _cached(path, fetch, is_stale=missing_polls)


def week_has_unfinished_games(games: pd.DataFrame) -> bool:
    """True if a cached week of games is empty or has any game not yet completed.

    A finished week never changes, so it is fetched once. A week still in progress (or
    not yet played) is re-fetched each time it is loaded, so new scores come in.
    """
    return games.empty or not bool(games["completed"].all())


def week_lines_not_final(lines: pd.DataFrame) -> bool:
    """True if cached week lines are empty or any game in them has no final score.

    Lines move until kickoff, so a week's lines are re-fetched until every game is over.
    """
    return lines.empty or bool(lines["homeScore"].isna().any())


def load_games_week(season: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Load one week of games involving an FBS team, for the season in progress.

    One API call the first time. The file data/raw/games_{season}_{season_type}_week_{WW}
    .parquet is re-fetched only while some game in it is not yet completed, so each
    finished week costs exactly one call (or two, if it was first loaded as a schedule).

    Args:
        season: Season year, e.g. 2026.
        week: Week number. Every postseason game is week 1.
        season_type: "regular" or "postseason".

    Returns:
        Raw games DataFrame (same CFBD columns as load_games), finished and unplayed games.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            games = cfbd.GamesApi(client).get_games(
                year=season,
                week=week,
                season_type=cfbd.SeasonType(season_type),
                classification=cfbd.DivisionClassification.FBS,
            )
        return pd.DataFrame(_to_records(games))

    path = config.RAW_DIR / f"games_{season}_{season_type}_week_{week:02d}.parquet"
    return _cached(path, fetch, is_stale=week_has_unfinished_games)


def load_lines_week(season: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Load one week of betting lines (all providers), for the season in progress.

    Re-fetched until every game in the week has a final score, so lines saved alongside
    a prediction are the latest available at that time. A benchmark only, never a
    model input.

    Args:
        season: Season year, e.g. 2026.
        week: Week number.
        season_type: "regular" or "postseason".

    Returns:
        One row per game x provider (see flatten_lines), CFBD sign convention.
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            lines = cfbd.BettingApi(client).get_lines(
                year=season, week=week, season_type=cfbd.SeasonType(season_type)
            )
        return flatten_lines(_to_records(lines))

    path = config.RAW_DIR / f"lines_{season}_{season_type}_week_{week:02d}.parquet"
    return _cached(path, fetch, is_stale=week_lines_not_final)


def load_advanced_game_stats_week(
    season: int, week: int, expected_games: int, season_type: str = "regular"
) -> pd.DataFrame:
    """Load one finished week of advanced game stats, for the season in progress.

    Call this only for weeks whose games are all completed. CFBD posts stats a little
    after games end, so a file covering fewer than 95% of `expected_games` is treated
    as incomplete and fetched again next time (historical coverage is 97.7-100%).

    Args:
        season: Season year, e.g. 2026.
        week: Week number.
        expected_games: Completed games involving an FBS team that week.
        season_type: "regular" or "postseason".

    Returns:
        Raw advanced game stats (same columns as load_advanced_game_stats).
    """

    def fetch() -> pd.DataFrame:
        with get_api_client() as client:
            stats = cfbd.StatsApi(client).get_advanced_game_stats(
                year=season,
                week=week,
                season_type=cfbd.SeasonType(season_type),
                exclude_garbage_time=True,
            )
        return flatten_records(_to_records(stats))

    def incomplete(stats: pd.DataFrame) -> bool:
        return stats.empty or stats["gameId"].nunique() < 0.95 * expected_games

    path = config.RAW_DIR / f"advanced_game_stats_{season}_{season_type}_week_{week:02d}.parquet"
    return _cached(path, fetch, is_stale=incomplete)
