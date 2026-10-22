"""Tune the Huber spread model; compare it with the ridge model, Elo, and the market.

Usage:
    python scripts/tune_spread_huber.py          # tune on config.TUNING_SEASONS
    python scripts/tune_spread_huber.py --test   # also score config.TEST_SEASON (run rarely)

The ridge model (spread.SpreadParams()) is left as tuned by scripts/tune_spread.py;
this script only chooses settings for the Huber model (spread.HUBER_PARAMS).

Reads data/processed/games.parquet; makes no API calls. Takes 10-20 minutes: each
setting reruns the full weekly walk-forward, and Huber refits several times per week.
"""

import argparse
import itertools

import pandas as pd

from football_model import clean, config, elo, evaluate, spread

# huber_c is in points of residual: results more than c points away from the
# prediction are discounted. If a best value lands on a grid edge, widen the grid
# in that direction and rerun.
GRID = {
    "huber_c": [7.0, 10.0, 14.0, 21.0, 28.0],
    "alpha": [1.5, 3.0, 6.0],
    "revert": [0.1, 0.2],
}


def grid_search(games: pd.DataFrame) -> pd.DataFrame:
    """Run the Huber walk-forward for every combination in GRID; score MAE on tuning seasons.

    Returns:
        One row per combination, best (lowest MAE) first.
    """
    in_tuning = games["season"].isin(config.TUNING_SEASONS)
    rows = []
    for huber_c, alpha, revert in itertools.product(*GRID.values()):
        params = spread.SpreadParams(alpha=alpha, revert=revert, huber_c=huber_c)
        pred = spread.run_spread(games, params)
        tuned = pred[in_tuning.values]
        score = evaluate.mae(tuned["margin"], tuned["spread_pred"])
        rows.append({"huber_c": huber_c, "alpha": alpha, "revert": revert, "mae": score})
        print(f"  {params} -> MAE {score:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def main() -> None:
    """Tune Huber, then report ridge vs. Huber vs. Elo vs. market on the tuning seasons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON].reset_index(drop=True)

    results = grid_search(games)
    results.to_csv(config.REPORTS_DIR / "spread_huber_tuning.csv", index=False)
    print("\nTop 10 Huber settings by tuning-season MAE:")
    print(results.head(10).round(4).to_string(index=False), "\n")

    best = results.iloc[0]
    huber_params = spread.SpreadParams(
        alpha=float(best["alpha"]), revert=float(best["revert"]), huber_c=float(best["huber_c"])
    )
    print(f"Best Huber: {huber_params}\n")

    # Every model on the same games, side by side. The hard 21-point margin cap is the
    # Phase 3 variant that failed; it is included to show where capping the margin hurts.
    ridge_params = spread.SpreadParams()
    capped_params = spread.SpreadParams(
        alpha=ridge_params.alpha, revert=ridge_params.revert, margin_cap=21.0
    )
    pred = spread.run_spread(games, ridge_params).rename(columns={"spread_pred": "ridge"})
    pred["huber"] = spread.run_spread(games, huber_params)["spread_pred"].to_numpy()
    pred["ridge_cap21"] = spread.run_spread(games, capped_params)["spread_pred"].to_numpy()
    rated = elo.predict(games, elo.EloParams(), config.TUNING_SEASONS)
    pred = pred.merge(rated[["game_id", "elo_spread"]], on="game_id")

    models = ["ridge", "huber", "ridge_cap21", "elo_spread"]
    print("MAE by part of season, tuning seasons:")
    print(evaluate.score_by_week(pred, models, config.TUNING_SEASONS).round(3).to_string(), "\n")
    print("MAE by |market spread|, tuning seasons:")
    print(evaluate.score_by_spread_size(pred, models, config.TUNING_SEASONS).round(3).to_string())

    if args.test:
        print(f"\nTEST season {config.TEST_SEASON}:")
        print(evaluate.score_by_week(pred, models, [config.TEST_SEASON]).round(3).to_string())


if __name__ == "__main__":
    main()
