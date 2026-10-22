"""Fetch and cache the Phase 5 data: advanced stats, recruiting, talent, returning production,
venues, and teams. Then report coverage and team-name mismatches.

Usage:
    python scripts/fetch_phase5_data.py --dry-run        # list missing files, no API calls
    python scripts/fetch_phase5_data.py --season 2024    # fetch one season (+ venues, teams)
    python scripts/fetch_phase5_data.py                  # fetch everything missing

Every fetch goes through data.py, which caches to data/raw/; files that already exist
cost no API calls. About 60 calls for a full first run.
"""

import argparse
from collections.abc import Callable

import pandas as pd

from football_model import clean, config, data

# Source name -> (loader, seasons, file prefix). Recruiting starts in 2010 so that
# 4-year recruiting averages exist for 2014. Preseason sources include the current
# season, since they are known before it starts; advanced stats stop at the last
# completed season.
LAST_COMPLETED = config.CURRENT_SEASON - 1
SEASONAL: dict[str, tuple[Callable[[int], pd.DataFrame], range]] = {
    "advanced_game_stats": (
        data.load_advanced_game_stats,
        range(config.FIRST_SEASON, LAST_COMPLETED + 1),
    ),
    "recruiting": (data.load_recruiting, range(2010, config.CURRENT_SEASON + 1)),
    "talent": (data.load_talent, range(config.FIRST_SEASON, config.CURRENT_SEASON + 1)),
    "returning_production": (
        data.load_returning_production,
        range(config.FIRST_SEASON, config.CURRENT_SEASON + 1),
    ),
}
STATIC: dict[str, Callable[[], pd.DataFrame]] = {
    "venues": data.load_venues,
    "teams": data.load_teams,
}


def planned_files(season: int | None) -> list[tuple[str, int | None]]:
    """Every (source, season) pair to fetch; season is None for static sources."""
    pairs = []
    for name, (_, seasons) in SEASONAL.items():
        pairs += [(name, s) for s in seasons if season is None or s == season]
    pairs += [(name, None) for name in STATIC]
    return pairs


def raw_path(name: str, season: int | None) -> str:
    """File name in data/raw/ for a source and season."""
    return f"{name}.parquet" if season is None else f"{name}_{season}.parquet"


def team_column(df: pd.DataFrame) -> str | None:
    """The column holding the school name in a CFBD table (team or school)."""
    for col in ["team", "school"]:
        if col in df.columns:
            return col
    return None


def coverage_report(games: pd.DataFrame) -> None:
    """Print rows per season per source, and FBS teams from the games table missing from each."""
    print("\nRows per season:")
    rows = {}
    for name, (loader, seasons) in SEASONAL.items():
        rows[name] = {s: len(loader(s)) for s in seasons}
    print(pd.DataFrame(rows).fillna(0).astype(int).to_string())

    print("\nFBS teams in the games table missing from each source (by season):")
    for name, (loader, seasons) in SEASONAL.items():
        for s in seasons:
            season_games = games[games["season"] == s]
            if season_games.empty:
                continue
            fbs = (set(season_games["home_team"]) | set(season_games["away_team"])) - {"FCS"}
            df = loader(s)
            col = team_column(df)
            have = set(df[col]) if col else set()
            missing = sorted(fbs - have)
            if missing:
                print(f"  {name} {s}: {len(missing)} missing: {', '.join(missing[:12])}")


def main() -> None:
    """Fetch missing files (or list them with --dry-run), then report coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List missing files only.")
    parser.add_argument("--season", type=int, help="Fetch only this season (+ static sources).")
    args = parser.parse_args()

    pairs = planned_files(args.season)
    missing = [(n, s) for n, s in pairs if not (config.RAW_DIR / raw_path(n, s)).exists()]
    print(f"{len(pairs)} files planned, {len(missing)} missing -> {len(missing)} API calls.")
    if args.dry_run:
        for n, s in missing:
            print(f"  would fetch {raw_path(n, s)}")
        return

    for n, s in missing:
        df = STATIC[n]() if s is None else SEASONAL[n][0](s)
        if df.empty:
            print(f"  NOTE: {raw_path(n, s)} is empty (CFBD has no data for it)")
    print(f"Done: {len(missing)} API calls made.")

    if args.season is None:
        coverage_report(clean.load_games_table())


if __name__ == "__main__":
    main()
