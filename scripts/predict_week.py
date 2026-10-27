"""Save this week's predictions to predictions/YYYY/week_WW.csv before kickoff.

Usage:
    python scripts/predict_week.py --week 9 --dry-run    # print, don't save
    python scripts/predict_week.py --week 9              # save CSV + Excel copy
    python scripts/predict_week.py --week 9 --allow-partial   # some games already started

Reads data/processed/games.parquet (completed seasons) plus the current season's weeks
from data/raw/. API calls: one per week not yet cached or still in progress, plus one
for the week's lines. Regular season only for now.
"""

import argparse

import pandas as pd

from football_model import clean, config, data, features, predict


def shadow_stack(
    history: pd.DataFrame, upcoming: pd.DataFrame, completed: pd.DataFrame, season: int, week: int
) -> pd.DataFrame | None:
    """Load the shadow stack's inputs and predict; on any problem, warn and return None.

    The shadow model must never stop the live predictions from being saved on time.
    API calls: one per finished week of this season's advanced stats not yet cached.
    """
    try:
        stats = [data.load_advanced_game_stats(s) for s in range(config.FIRST_SEASON, season)]
        for w in range(1, week):
            expected = int((completed["week"] == w).sum())
            if expected:
                stats.append(data.load_advanced_game_stats_week(season, w, expected))
        return predict.stack_shadow(
            history,
            upcoming,
            pd.concat(stats, ignore_index=True),
            data.load_returning_production(season),
            data.load_teams(),
            data.load_venues(),
            features.load_features(),
        )
    except Exception as err:  # noqa: BLE001 - report and carry on without the shadow model
        print(f"WARNING: shadow stack skipped ({type(err).__name__}: {err})")
        return None


def main() -> None:
    """Build predictions for one week, print a summary, and save unless --dry-run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True, help="Regular-season week to predict.")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving.")
    parser.add_argument(
        "--allow-partial", action="store_true", help="Save only games not yet kicked off."
    )
    args = parser.parse_args()

    season = config.CURRENT_SEASON
    now = pd.Timestamp.now(tz="UTC")

    past = clean.load_games_table()
    past = past[past["season"] < season]
    completed, upcoming = clean.build_current_season(season, args.week)
    history = pd.concat([past, completed], ignore_index=True)

    if upcoming.empty:
        raise SystemExit(f"No unplayed games found for {season} week {args.week}.")

    played = completed[completed["week"] == args.week]
    print(f"\n{season} season so far: {len(completed)} completed games, weeks 1-{args.week}.")
    print(f"Week {args.week}: {len(upcoming)} unplayed, {len(played)} already final.")
    print(f"Kickoffs (UTC): {upcoming['start_date'].min()} to {upcoming['start_date'].max()}")
    print(f"Now (UTC):      {now}")
    print(f"Games with a market line: {int(upcoming['market_spread'].notna().sum())}\n")

    stack = shadow_stack(history, upcoming, completed, season, args.week)
    preds = predict.build_week_predictions(history, upcoming, created_at=now, stack=stack)
    if args.allow_partial:
        preds = predict.drop_started(preds, now)

    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(predict.readable_table(preds).to_string(index=False))
    print(
        f"\nsigma (ridge) = {preds['sigma'].iloc[0]:.2f}, games in fit = {preds['games_in_fit'].iloc[0]}"
    )
    has_line = preds["market_spread"].notna()
    if has_line.any():
        diff = (preds.loc[has_line, "ridge_spread"] - preds.loc[has_line, "market_spread"]).abs()
        corr = preds.loc[has_line, ["ridge_spread", "market_spread"]].corr().iloc[0, 1]
        print(f"Ridge vs. market: mean |diff| {diff.mean():.2f} points, correlation {corr:.3f}")
    if preds["stack_spread"].notna().any():
        gap = (preds["stack_spread"] - preds["ridge_spread"]).abs()
        corr = preds[["stack_spread", "ridge_spread"]].corr().iloc[0, 1]
        print(
            f"Shadow stack vs. ridge: mean |diff| {gap.mean():.2f} points, correlation {corr:.3f}"
            " (recorded in the CSV and Excel, not the live pick)"
        )

    path = predict.prediction_path(season, args.week)
    if args.dry_run:
        print(f"\nDry run: nothing saved (would write {path}).")
        return
    predict.save_predictions(preds, path, now)
    print(f"\nSaved {len(preds)} predictions to {path}")
    xlsx = predict.excel_path(season, args.week)
    predict.write_excel(preds, xlsx)
    print(f"Saved readable copy (one tab per conference) to {xlsx}")
    print(f"\n  {path.name} is the permanent record: commit it, and never edit or delete it.")
    print(f"  {xlsx.name} is the copy to read; rebuild it any time with")
    print(f"  python scripts/export_predictions.py --week {args.week}")


if __name__ == "__main__":
    main()
