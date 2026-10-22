"""Tune the ridge spread model by grid search on the tuning seasons; compare with Elo and the market.

Usage:
    python scripts/tune_spread.py          # tune on config.TUNING_SEASONS
    python scripts/tune_spread.py --test   # also score config.TEST_SEASON (run rarely)

Reads data/processed/games.parquet; makes no API calls. Takes several minutes: every
setting reruns the full weekly walk-forward (about 5 seconds each).
"""

import argparse
import itertools

import pandas as pd

from football_model import clean, config, elo, evaluate, spread

# Two stages, to keep the run short. Stage 1 finds alpha and revert with no margin cap
# and equal weights. Stage 2 tries caps and recency weights, re-tuning alpha near the
# stage-1 best because down-weighting old games makes the penalty relatively stronger.
# If a best value lands on a grid edge, widen the grid in that direction and rerun.
STAGE_1 = {
    "alpha": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    "revert": [0.0, 0.1, 0.2, 0.3, 0.45],
    "margin_cap": [None],
    "half_life_weeks": [None],
}
STAGE_2_ALPHA_MULTIPLIERS = [0.5, 1.0, 2.0]
STAGE_2 = {
    "margin_cap": [None, 21.0, 28.0, 35.0, 42.0],
    "half_life_weeks": [None, 4.0, 8.0, 16.0],
}


def evaluate_grid(games: pd.DataFrame, grid: dict[str, list]) -> pd.DataFrame:
    """Run the walk-forward model for every combination in `grid`; score MAE on tuning seasons.

    Returns:
        One row per combination, best (lowest MAE) first.
    """
    in_tuning = games["season"].isin(config.TUNING_SEASONS)
    rows = []
    for values in itertools.product(*grid.values()):
        params = spread.SpreadParams(**dict(zip(grid, values)))
        pred = spread.run_spread(games, params)
        tuned = pred[in_tuning.values]
        rows.append(
            {**dict(zip(grid, values)), "mae": evaluate.mae(tuned["margin"], tuned["spread_pred"])}
        )
        print(f"  {params} -> MAE {rows[-1]['mae']:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def params_from_row(row: pd.Series) -> spread.SpreadParams:
    """Turn a results row back into SpreadParams (pandas stores None as NaN)."""
    cap = None if pd.isna(row["margin_cap"]) else float(row["margin_cap"])
    half_life = None if pd.isna(row["half_life_weeks"]) else float(row["half_life_weeks"])
    return spread.SpreadParams(
        alpha=float(row["alpha"]),
        revert=float(row["revert"]),
        margin_cap=cap,
        half_life_weeks=half_life,
    )


def main() -> None:
    """Tune, report ridge vs. Elo vs. market on the tuning seasons, and optionally score the test season."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON].reset_index(drop=True)

    # The games table is already sorted by (start_date, game_id), the same order
    # run_spread returns, so the tuning-season mask lines up row for row.
    print("Stage 1: alpha x revert")
    stage_1 = evaluate_grid(games, STAGE_1)
    best_1 = stage_1.iloc[0]

    print("Stage 2: margin cap x recency half-life (alpha near stage-1 best)")
    stage_2_grid = {
        "alpha": [best_1["alpha"] * m for m in STAGE_2_ALPHA_MULTIPLIERS],
        "revert": [best_1["revert"]],
        **STAGE_2,
    }
    stage_2 = evaluate_grid(games, stage_2_grid)

    results = pd.concat([stage_1.assign(stage=1), stage_2.assign(stage=2)])
    results = results.sort_values("mae").reset_index(drop=True)
    results.to_csv(config.REPORTS_DIR / "spread_tuning.csv", index=False)
    print("\nTop 10 settings by tuning-season MAE:")
    print(results.head(10).round(4).to_string(index=False), "\n")

    params = params_from_row(results.iloc[0])
    print(f"Best: {params}\n")

    pred = spread.run_spread(games, params)
    rated = elo.predict(games, elo.EloParams(), config.TUNING_SEASONS)
    pred = pred.merge(rated[["game_id", "elo_spread"]], on="game_id")

    table = evaluate.score_seasons(pred, None, "spread_pred", config.TUNING_SEASONS)
    print("Ridge vs. market, tuning seasons (games with a line):")
    print(table.round(3).to_string(), "\n")

    by_week = evaluate.score_by_week(pred, ["spread_pred", "elo_spread"], config.TUNING_SEASONS)
    print("MAE by part of season, tuning seasons:")
    print(by_week.round(3).to_string(), "\n")

    final = spread.final_ratings(games, params, config.TEST_SEASON - 1)
    print(f"Top 10, end of {config.TEST_SEASON - 1} (home field {final.attrs['home_field']:.2f}):")
    print(final.head(10).round(1).to_string(index=False))

    if args.test:
        test = evaluate.score_by_week(pred, ["spread_pred", "elo_spread"], [config.TEST_SEASON])
        print(f"\nTEST season {config.TEST_SEASON}:")
        print(test.round(3).to_string())


if __name__ == "__main__":
    main()
