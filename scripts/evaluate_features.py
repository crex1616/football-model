"""Phase 5b: does each feature group improve walk-forward log loss?

Usage:
    python scripts/evaluate_features.py            # tune, build features, score
    python scripts/evaluate_features.py --cached   # reuse data/processed/features.parquet
    python scripts/evaluate_features.py --test     # also score the 2025 test season (rarely)

Steps:
1. Tune the efficiency ratings (alpha, revert) by how well they predict unseen games'
   EPA per play and success rate, on the tuning seasons.
2. Build every feature (features.build_features) and save it.
3. Add feature groups one at a time to the stacked ridge model (ratings, then
   efficiency, preseason, situational). Each step's penalty is tuned on the scored
   seasons; a group is kept only if it lowers log loss.
4. Print every step vs. the Phase 4 baseline and the market on the same games, overall
   and for weeks 1-4, plus the final model's coefficient table and calibration plot.

Scored seasons: 2016-2024 excluding 2020 (the stack needs an earlier season to train
on). No API calls once the Phase 5 data is cached.
"""

import argparse

import pandas as pd

from football_model import clean, config, data, evaluate, features, models, win_prob

SCORED_SEASONS = [s for s in config.TUNING_SEASONS if s >= 2016]
EFFICIENCY_ALPHAS = [1.0, 3.0, 10.0, 30.0]
EFFICIENCY_REVERTS = [0.1, 0.3, 0.5]
STACK_ALPHAS = [1.0, 10.0, 100.0, 1000.0]


def tune_and_build(games: pd.DataFrame) -> pd.DataFrame:
    """Tune the efficiency ratings, then build and save the feature table."""
    seasons = sorted(games["season"].unique())
    stats = pd.concat([data.load_advanced_game_stats(s) for s in seasons], ignore_index=True)
    with_stats = games.merge(features.game_efficiency(stats, games), on="game_id", how="left")

    best = {}
    for metric in features.METRICS:
        table = features.tune_efficiency(
            with_stats, metric, EFFICIENCY_ALPHAS, EFFICIENCY_REVERTS, config.TUNING_SEASONS
        )
        print(f"Efficiency tuning, {metric} (MSE of pre-game prediction, tuning seasons):")
        print(table.head(5).round(5).to_string(index=False), "\n")
        top = table.iloc[0]
        best[metric] = features.EfficiencyParams(float(top["alpha"]), float(top["revert"]))

    built = features.build_features(games, best)
    features.save_features(built)
    print(f"Saved {len(built):,} games of features to {config.FEATURES_TABLE_PATH}\n")
    return built


def score_step(frame: pd.DataFrame, prob_col: str, spread_col: str, seasons: list[int]) -> dict:
    """Log loss, Brier, MAE overall and log loss in weeks 1-4, model and market."""
    overall = evaluate.score_seasons(frame, prob_col, spread_col, seasons).loc["all"]
    by_week = evaluate.score_probs_by_week(frame, [prob_col], seasons)
    return {
        "log_loss": overall["model_log_loss"],
        "brier": overall["model_brier"],
        "mae": overall["model_mae"],
        "log_loss_wk1_4": by_week.loc["weeks 1-4", prob_col],
        "market_log_loss": overall["market_log_loss"],
        "market_brier": overall["market_brier"],
        "market_mae": overall["market_mae"],
        "market_wk1_4": by_week.loc["weeks 1-4", "market"],
        "games": int(overall["games"]),
    }


def run_stack(
    frame: pd.DataFrame, cols: list[str], seasons: list[int]
) -> tuple[float, pd.DataFrame]:
    """Tune the stack's penalty on `seasons`; return the best alpha and its predictions."""
    best = None
    for alpha in STACK_ALPHAS:
        pred = models.walk_forward_stacked(frame, cols, alpha, seasons)
        scored = frame.merge(pred, on="game_id")
        loss = evaluate.score_seasons(scored, "stack_prob", None, seasons).loc[
            "all", "model_log_loss"
        ]
        if best is None or loss < best[0]:
            best = (loss, alpha, scored)
    return best[1], best[2]


def main() -> None:
    """Tune, build, and score every feature group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true", help="Reuse the saved features.")
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON].reset_index(drop=True)
    frame = features.load_features() if args.cached else tune_and_build(games)

    # Phase 4 baseline: ridge spread, Phi(spread / sigma) with walk-forward sigma.
    frame["baseline_prob"] = win_prob.normal_prob(frame, "ridge_spread")
    frame = evaluate.add_market_prob(frame, config.TUNING_SEASONS)

    rows = {
        "baseline (Phase 4)": score_step(frame, "baseline_prob", "ridge_spread", SCORED_SEASONS)
    }
    kept: list[str] = []
    best_loss = None
    final = None
    for group, cols in features.FEATURE_GROUPS.items():
        trial = kept + cols
        alpha, scored = run_stack(frame, trial, SCORED_SEASONS)
        result = score_step(scored, "stack_prob", "stack_spread", SCORED_SEASONS)
        improved = best_loss is None or result["log_loss"] < best_loss
        label = f"+ {group}" if kept else f"stack: {group}"
        rows[f"{label} (alpha {alpha:g}){'' if improved else '  [dropped]'}"] = result
        if improved:
            kept, best_loss, final = trial, result["log_loss"], (alpha, scored)

    table = pd.DataFrame.from_dict(rows, orient="index")
    market = table.iloc[0][["market_log_loss", "market_brier", "market_mae", "market_wk1_4"]]
    table = table[["games", "log_loss", "brier", "mae", "log_loss_wk1_4"]]
    table.loc["market (closing line)"] = [
        table.iloc[0]["games"],
        *market.to_numpy(),
    ]
    seasons = f"{SCORED_SEASONS[0]}-{SCORED_SEASONS[-1]} excl. 2020"
    print(f"Walk-forward scores, {seasons}, games with a line:")
    print(table.round(4).to_string(), "\n")
    print(f"Kept features: {kept}\n")

    alpha, scored = final
    train = models.training_rows(frame, config.TEST_SEASON)
    train = train[train["season"] <= max(SCORED_SEASONS)]
    print("Coefficients (OLS on standardized features, seasons 2015-2024 excl. 2020):")
    print(models.coefficient_table(train, kept).round(3).to_string(), "\n")

    in_scored = scored[scored["season"].isin(SCORED_SEASONS)]
    path = config.FIGURES_DIR / "stack_calibration.png"
    evaluate.calibration_plot(
        in_scored["home_win"], in_scored["stack_prob"], path, title=f"Stacked model, {seasons}"
    )
    print(f"Saved calibration plot to {path}")

    if args.test:
        test = [config.TEST_SEASON]
        pred = models.walk_forward_stacked(frame, kept, alpha, test)
        test_scored = frame.merge(pred, on="game_id")
        print(f"\nTEST season {config.TEST_SEASON}:")
        print("stack:   ", score_step(test_scored, "stack_prob", "stack_spread", test))
        print("baseline:", score_step(frame, "baseline_prob", "ridge_spread", test))


if __name__ == "__main__":
    main()
