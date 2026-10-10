"""Tune Elo hyperparameters by grid search on the tuning seasons, and compare with the market.

Usage:
    python scripts/tune_elo.py          # tune on config.TUNING_SEASONS
    python scripts/tune_elo.py --test   # also score config.TEST_SEASON (run rarely)

Reads data/processed/games.parquet; makes no API calls.
"""

import argparse
import itertools

import pandas as pd

from football_model import clean, config, elo, evaluate

# A first, coarser grid (k 10-50, home_field 40-85, revert 0.2-0.5) put the best
# settings at revert = 0.2, its lower edge, so the grid was widened there. If a best
# value lands on an edge, widen the grid in that direction and rerun.
GRID = {
    "k": [10, 20, 25, 30, 35, 40, 50],
    "home_field": [40, 50, 55, 60, 70],
    "revert": [0.0, 0.1, 0.15, 0.2, 0.33],
    "use_mov": [True, False],
}


def grid_search(games: pd.DataFrame) -> pd.DataFrame:
    """Run Elo for every combination in GRID; score log loss on the tuning seasons.

    Returns:
        One row per combination, sorted best (lowest log loss) first.
    """
    in_tuning = games["season"].isin(config.TUNING_SEASONS)
    rows = []
    for k, home_field, revert, use_mov in itertools.product(*GRID.values()):
        params = elo.EloParams(k=k, home_field=home_field, revert=revert, use_mov=use_mov)
        rated = elo.run_elo(games, params)
        tuned = rated[in_tuning.values]
        rows.append(
            {
                "k": k,
                "home_field": home_field,
                "revert": revert,
                "use_mov": use_mov,
                "log_loss": evaluate.log_loss(tuned["home_win"], tuned["elo_prob_home"]),
                "brier": evaluate.brier_score(tuned["home_win"], tuned["elo_prob_home"]),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def main() -> None:
    """Tune, report model vs. market on the tuning seasons, and optionally score the test season."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON]
    games = evaluate.add_market_prob(games, config.TUNING_SEASONS)

    results = grid_search(games)
    results.to_csv(config.REPORTS_DIR / "elo_tuning.csv", index=False)
    print("Top 10 settings by tuning-season log loss:")
    print(results.head(10).round(4).to_string(index=False), "\n")

    best = results.iloc[0]
    params = elo.EloParams(
        k=float(best["k"]),
        home_field=float(best["home_field"]),
        revert=float(best["revert"]),
        use_mov=bool(best["use_mov"]),
    )
    print(f"Best: {params}\n")

    rated = elo.predict(games, params, config.TUNING_SEASONS)
    table = evaluate.score_seasons(rated, "elo_prob_home", "elo_spread", config.TUNING_SEASONS)
    print("Elo vs. market, tuning seasons (games with a line):")
    print(table.round(4).to_string(), "\n")

    tuning = rated[rated["season"].isin(config.TUNING_SEASONS)]
    evaluate.calibration_plot(
        tuning["home_win"],
        tuning["elo_prob_home"],
        config.FIGURES_DIR / "elo_calibration.png",
        title=f"Elo calibration, {min(config.TUNING_SEASONS)}-{max(config.TUNING_SEASONS)}",
    )
    print(f"Saved calibration plot to {config.FIGURES_DIR / 'elo_calibration.png'}")

    if args.test:
        test = evaluate.score_seasons(rated, "elo_prob_home", "elo_spread", [config.TEST_SEASON])
        print(f"\nTEST season {config.TEST_SEASON}:")
        print(test.round(4).to_string())


if __name__ == "__main__":
    main()
