"""Score win-probability models (ridge and Huber spreads) against Elo and the market.

Usage:
    python scripts/evaluate_win_prob.py          # score config.TUNING_SEASONS
    python scripts/evaluate_win_prob.py --test   # also score config.TEST_SEASON (run rarely)

Reads data/processed/games.parquet; makes no API calls. Saves calibration plots to
reports/figures/. Takes about a minute (two walk-forward spread runs).
"""

import argparse

import pandas as pd

from football_model import clean, config, elo, evaluate, spread, win_prob


def build_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Spreads and win probabilities from every model, plus Elo and the market, on one frame."""
    pred = spread.run_spread(games, spread.SpreadParams()).rename(columns={"spread_pred": "ridge"})
    pred["huber"] = spread.run_spread(games, spread.HUBER_PARAMS)["spread_pred"].to_numpy()

    rated = elo.predict(games, elo.EloParams(), config.TUNING_SEASONS)
    pred = pred.merge(rated[["game_id", "elo_prob_home"]], on="game_id")
    pred = evaluate.add_market_prob(pred, config.TUNING_SEASONS)

    pred = win_prob.add_win_probs(pred, "ridge", "ridge")
    return win_prob.add_win_probs(pred, "huber", "huber")


def summary_table(pred: pd.DataFrame, prob_cols: list[str], seasons: list[int]) -> pd.DataFrame:
    """One row per model: log loss and Brier on the given seasons, market and Elo included."""
    rows = {}
    for col in prob_cols:
        overall = evaluate.score_seasons(pred, col, None, seasons).loc["all"]
        rows[col] = {"log_loss": overall["model_log_loss"], "brier": overall["model_brier"]}
    rows["market"] = {"log_loss": overall["market_log_loss"], "brier": overall["market_brier"]}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("log_loss")


def main() -> None:
    """Score every probability model, print the logistic cross-check, save calibration plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON].reset_index(drop=True)
    pred = build_predictions(games)

    prob_cols = [
        f"{model}_prob_{method}"
        for model in ["ridge", "huber"]
        for method in ["normal", "normal_by_week", "logistic"]
    ] + ["elo_prob_home"]

    print("Log loss and Brier, tuning seasons (games with a line):")
    print(summary_table(pred, prob_cols, config.TUNING_SEASONS).round(4).to_string(), "\n")

    print("Log loss by part of season, tuning seasons:")
    by_week = evaluate.score_probs_by_week(pred, prob_cols, config.TUNING_SEASONS)
    print(by_week.round(4).to_string(), "\n")

    # Cross-check: free logistic fit on the tuning seasons vs. the implied probit slope.
    sigma = win_prob.residual_sd(pred, "ridge", config.TUNING_SEASONS)
    logit = win_prob.fit_logistic(pred, "ridge", config.TUNING_SEASONS)
    print(f"Ridge residual SD, tuning seasons: {sigma:.2f} points")
    print(f"Logistic: intercept {logit.params['const']:.3f}, slope {logit.params['ridge']:.4f}")
    print(f"Slope implied by Phi(spread / sigma): about 1.7 / sigma = {1.7 / sigma:.4f}\n")

    tuning = pred[pred["season"].isin(config.TUNING_SEASONS)]
    for col in prob_cols:
        path = config.FIGURES_DIR / f"win_prob_calibration_{col}.png"
        evaluate.calibration_plot(
            tuning["home_win"], tuning[col], path, title=f"{col}, tuning seasons"
        )
    print(f"Saved calibration plots to {config.FIGURES_DIR}")

    if args.test:
        print(f"\nTEST season {config.TEST_SEASON}:")
        print(summary_table(pred, prob_cols, [config.TEST_SEASON]).round(4).to_string())


if __name__ == "__main__":
    main()
