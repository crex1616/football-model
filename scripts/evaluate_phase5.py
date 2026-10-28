"""Phase 5c: compare the stack, penalized logistic, and LightGBM on the same games.

Usage:
    python scripts/evaluate_phase5.py           # tune and score on the tuning seasons
    python scripts/evaluate_phase5.py --test    # also score the 2025 test season (rarely)

Every model uses the same features (models.STACK_FEATURES, all four groups from 5b) and
the same walk-forward split: to predict season N, train on seasons before N (no burn-in,
no 2020). Feature groups are not re-selected per model; only each model's own
hyperparameters are tuned, on the scored seasons.

Prints, for baseline (Phase 4), Elo, stack, logistic, LightGBM, and the market on the
same games: log loss, Brier, MAE, and log loss in weeks 1-4; then each model's paired
log-loss difference vs. the baseline with a standard error, the logistic coefficient
table, LightGBM's feature importance, and calibration plots.

Reads data/processed/features.parquet (built by scripts/evaluate_features.py). No API
calls.
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from football_model import config, evaluate, features, models, win_prob

SCORED_SEASONS = [s for s in config.TUNING_SEASONS if s >= 2016]
LOGIT_CS = [0.001, 0.01, 0.1, 1.0, 10.0]
LGBM_GRID = [
    models.LgbmParams(leaves, rate, trees, min_child)
    for leaves, rate, trees, min_child in itertools.product(
        [2, 4, 8], [0.01, 0.02], [100, 200, 400], [50]
    )
]
# A first grid (leaves 4-16, rate 0.02-0.05, 200-500 trees) was best at its simplest
# corner, so the grid was moved toward simpler models. Two leaves = one split per tree:
# an additive model with no interactions. A one-off check of stumps with 800 and 1,200
# trees (rate 0.02) gave 0.4977 and 0.4983: the same plateau as the best grid point.


def log_loss_on(frame: pd.DataFrame, prob_col: str, seasons: list[int]) -> float:
    """Overall log loss of prob_col on games with a line in `seasons`."""
    return evaluate.score_seasons(frame, prob_col, None, seasons).loc["all", "model_log_loss"]


def tune_logistic(frame: pd.DataFrame, cols: list[str]) -> tuple[float, pd.DataFrame]:
    """Pick C by walk-forward log loss; return it and the merged predictions."""
    print("Logistic tuning (log loss, scored seasons):")
    best = None
    for C in LOGIT_CS:
        scored = frame.merge(
            models.walk_forward_logistic(frame, cols, C, SCORED_SEASONS), on="game_id"
        )
        loss = log_loss_on(scored, "logit_prob", SCORED_SEASONS)
        print(f"  C = {C:<6g} {loss:.4f}")
        if best is None or loss < best[0]:
            best = (loss, C, scored)
    print()
    return best[1], best[2]


def tune_lgbm(frame: pd.DataFrame, cols: list[str]) -> tuple[models.LgbmParams, pd.DataFrame]:
    """Pick LightGBM settings by walk-forward log loss; return them and the predictions."""
    print("LightGBM tuning (log loss and MAE, scored seasons):")
    best = None
    for params in LGBM_GRID:
        scored = frame.merge(
            models.walk_forward_lgbm(frame, cols, params, SCORED_SEASONS), on="game_id"
        )
        table = evaluate.score_seasons(scored, "lgbm_prob", "lgbm_spread", SCORED_SEASONS)
        loss, mae = table.loc["all", ["model_log_loss", "model_mae"]]
        print(
            f"  leaves {params.num_leaves:>2}, rate {params.learning_rate:<4g}, "
            f"trees {params.n_estimators:>3}: {loss:.4f}  MAE {mae:.3f}"
        )
        if best is None or loss < best[0]:
            best = (loss, params, scored)
    print()
    return best[1], best[2]


def score_row(
    frame: pd.DataFrame, prob_col: str, spread_col: str | None, seasons: list[int]
) -> dict:
    """Games, log loss, Brier, MAE (if a spread exists), and log loss in weeks 1-4."""
    overall = evaluate.score_seasons(frame, prob_col, spread_col, seasons).loc["all"]
    by_week = evaluate.score_probs_by_week(frame, [prob_col], seasons)
    return {
        "games": int(overall["games"]),
        "log_loss": overall["model_log_loss"],
        "brier": overall["model_brier"],
        "mae": overall["model_mae"] if spread_col else np.nan,
        "log_loss_wk1_4": by_week.loc["weeks 1-4", prob_col],
    }


def market_row(frame: pd.DataFrame, seasons: list[int]) -> dict:
    """The closing line's scores on the same games."""
    overall = evaluate.score_seasons(frame, "baseline_prob", "ridge_spread", seasons).loc["all"]
    by_week = evaluate.score_probs_by_week(frame, ["baseline_prob"], seasons)
    return {
        "games": int(overall["games"]),
        "log_loss": overall["market_log_loss"],
        "brier": overall["market_brier"],
        "mae": overall["market_mae"],
        "log_loss_wk1_4": by_week.loc["weeks 1-4", "market"],
    }


def per_game_log_loss(y: pd.Series, p: pd.Series) -> pd.Series:
    """Each game's log loss, clipped like evaluate.log_loss."""
    p = p.clip(1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_difference(
    frame: pd.DataFrame, prob_cols: list[str], seasons: list[int]
) -> pd.DataFrame:
    """Per-game log loss of each model minus the baseline: mean, SE = sd / sqrt(n), t.

    The same games are scored by both models, so differencing game by game removes most
    of the noise that comes from which games happened to be played (a paired t-test).
    """
    games = frame[frame["market_spread"].notna() & frame["season"].isin(seasons)]
    y = games["home_win"].astype(float)
    base = per_game_log_loss(y, games["baseline_prob"])
    rows = {}
    for col in prob_cols:
        diff = per_game_log_loss(y, games[col]) - base
        se = diff.std() / np.sqrt(len(diff))
        rows[col] = {"mean_diff": diff.mean(), "se": se, "t": diff.mean() / se}
    return pd.DataFrame.from_dict(rows, orient="index")


def comparison_table(frame: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """One row per model plus the market, all on the same games."""
    rows = {
        "baseline (Phase 4)": score_row(frame, "baseline_prob", "ridge_spread", seasons),
        "Elo": score_row(frame, "elo_prob", None, seasons),
        "stack (ridge)": score_row(frame, "stack_prob", "stack_spread", seasons),
        "logistic": score_row(frame, "logit_prob", None, seasons),
        "LightGBM": score_row(frame, "lgbm_prob", "lgbm_spread", seasons),
        "market (closing line)": market_row(frame, seasons),
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def predict_all(
    frame: pd.DataFrame, cols: list[str], C: float, params: models.LgbmParams, seasons: list[int]
) -> pd.DataFrame:
    """Add stack, logistic, and LightGBM predictions for `seasons` to the frame."""
    stack = models.walk_forward_stacked(frame, cols, models.STACK_ALPHA, seasons)
    logit = models.walk_forward_logistic(frame, cols, C, seasons)
    lgbm = models.walk_forward_lgbm(frame, cols, params, seasons)
    return frame.merge(stack, on="game_id").merge(logit, on="game_id").merge(lgbm, on="game_id")


def main() -> None:
    """Tune, score, and compare every Phase 5 model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    frame = features.load_features()
    frame = frame[frame["season"] <= config.TEST_SEASON].reset_index(drop=True)
    frame["baseline_prob"] = win_prob.normal_prob(frame, "ridge_spread")
    # elo_diff already includes home field (zero at neutral sites).
    frame["elo_prob"] = 1.0 / (1.0 + 10.0 ** (-frame["elo_diff"] / 400.0))
    frame = evaluate.add_market_prob(frame, config.TUNING_SEASONS)
    cols = models.STACK_FEATURES

    C, _ = tune_logistic(frame, cols)
    params, _ = tune_lgbm(frame, cols)
    scored = predict_all(frame, cols, C, params, SCORED_SEASONS)

    seasons = f"{SCORED_SEASONS[0]}-{SCORED_SEASONS[-1]} excl. 2020"
    print(f"Walk-forward scores, {seasons}, games with a line:")
    print(f"(stack alpha {models.STACK_ALPHA:g}; logistic C {C:g}; LightGBM {params})")
    print(comparison_table(scored, SCORED_SEASONS).round(4).to_string(), "\n")

    prob_cols = ["elo_prob", "stack_prob", "logit_prob", "lgbm_prob"]
    print("Per-game log loss minus baseline (negative = better than baseline):")
    print(paired_difference(scored, prob_cols, SCORED_SEASONS).round(4).to_string(), "\n")

    sigmas = scored.groupby("season")["lgbm_sigma"].first()
    stack_sigma = models.fit_stacked(
        models.training_rows(frame, config.TEST_SEASON), cols, models.STACK_ALPHA
    )[1]
    print("LightGBM sigma by season (held-out last training season):")
    print(sigmas.round(2).to_string())
    print(f"(stack in-sample sigma, trained through 2024: {stack_sigma:.2f})\n")

    train = models.training_rows(frame, config.TEST_SEASON)
    print("Logistic coefficients (unpenalized logit, standardized, 2015-2024 excl. 2020):")
    print(models.logit_coefficient_table(train, cols).round(3).to_string(), "\n")

    lgbm_model = models.fit_lgbm(train, cols, params)
    print("LightGBM feature importance (share of total gain, trained 2015-2024):")
    print(models.lgbm_importance(lgbm_model, cols).round(3).to_string(), "\n")

    in_scored = scored[scored["season"].isin(SCORED_SEASONS)]
    for col, name, file in [
        ("logit_prob", "Penalized logistic", "logit_calibration.png"),
        ("lgbm_prob", "LightGBM", "lgbm_calibration.png"),
    ]:
        path = config.FIGURES_DIR / file
        evaluate.calibration_plot(
            in_scored["home_win"], in_scored[col], path, title=f"{name}, {seasons}"
        )
        print(f"Saved calibration plot to {path}")

    if args.test:
        test = [config.TEST_SEASON]
        test_scored = predict_all(frame, cols, C, params, test)
        print(f"\nTEST season {config.TEST_SEASON}, games with a line:")
        print(comparison_table(test_scored, test).round(4).to_string(), "\n")
        print("Per-game log loss minus baseline:")
        print(paired_difference(test_scored, prob_cols, test).round(4).to_string())


if __name__ == "__main__":
    main()
