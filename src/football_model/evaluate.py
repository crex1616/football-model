"""Metrics and model-vs-market scoring shared by every phase.

Probability models are judged by log loss and Brier score; spread models by mean
absolute error. Every score is shown next to the closing market line on the same
games, so progress is always measured as "model vs. market."
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import metrics


def log_loss(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    """Mean log loss of predicted probabilities p for 0/1 outcomes y. Lower is better.

    This is the negative mean Bernoulli log-likelihood, the same objective a logit
    maximizes. A coin flip (p = 0.5 always) scores ln 2 = 0.693. It punishes
    confident misses heavily, so it rewards honest (calibrated) probabilities.
    """
    return float(metrics.log_loss(y, p, labels=[0, 1]))


def brier_score(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    """Mean squared error of probabilities: mean((p - y)^2). Lower is better; 0.25 = coin flip.

    This is the residual sum of squares of a linear probability model, scored out of sample.
    """
    return float(metrics.brier_score_loss(y, p))


def mae(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> float:
    """Mean absolute error in points (used for spreads)."""
    return float(metrics.mean_absolute_error(y, yhat))


def market_residual_sd(games: pd.DataFrame, seasons: Iterable[int]) -> float:
    """Standard deviation of (margin - market_spread) on the given seasons.

    This is sigma for turning a spread into a win probability. Estimate it on
    training seasons only, then apply it to later seasons.

    Args:
        games: Games table with margin and market_spread.
        seasons: Seasons to estimate on.

    Returns:
        Residual standard deviation in points (about 15.5).
    """
    subset = games[games["season"].isin(list(seasons)) & games["market_spread"].notna()]
    return float((subset["margin"] - subset["market_spread"]).std())


def market_win_prob(market_spread: np.ndarray | pd.Series, sigma: float) -> np.ndarray:
    """Convert market spreads (expected home margin) to home win probabilities.

    Assumes the actual margin is Normal(spread, sigma), so P(home win) =
    P(margin > 0) = Phi(spread / sigma). This is a probit link with the slope fixed
    at 1 / sigma. A spread of 0 gives 0.5.

    Args:
        market_spread: Expected home margin (our sign convention).
        sigma: Residual standard deviation, from market_residual_sd.

    Returns:
        Array of probabilities (NaN where the spread is missing).
    """
    return norm.cdf(np.asarray(market_spread, dtype=float) / sigma)


def add_market_prob(games: pd.DataFrame, fit_seasons: Iterable[int]) -> pd.DataFrame:
    """Add market_prob_home, using sigma estimated on fit_seasons only.

    Args:
        games: Games table with margin and market_spread.
        fit_seasons: Seasons used to estimate sigma (training only).

    Returns:
        New DataFrame with a market_prob_home column.
    """
    sigma = market_residual_sd(games, fit_seasons)
    games = games.copy()
    games["market_prob_home"] = market_win_prob(games["market_spread"], sigma)
    return games


def score_seasons(
    games: pd.DataFrame,
    prob_col: str,
    spread_col: str | None = None,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Score a model and the market, per season and overall, on the same games.

    Only games that have a market line are scored, so the comparison is fair.

    Args:
        games: Must contain season, home_win, margin, market_spread,
            market_prob_home, and the model's prob_col (and spread_col if given).
        prob_col: Column with the model's home win probability.
        spread_col: Optional column with the model's expected home margin.
        seasons: Seasons to score; all seasons in `games` if None.

    Returns:
        One row per season plus an "all" row. Columns: games, model_log_loss,
        market_log_loss, model_brier, market_brier, and (with spread_col) model_mae,
        market_mae.
    """
    scored = games[games["market_spread"].notna()]
    if seasons is not None:
        scored = scored[scored["season"].isin(list(seasons))]

    def one_block(block: pd.DataFrame) -> dict:
        row = {
            "games": len(block),
            "model_log_loss": log_loss(block["home_win"], block[prob_col]),
            "market_log_loss": log_loss(block["home_win"], block["market_prob_home"]),
            "model_brier": brier_score(block["home_win"], block[prob_col]),
            "market_brier": brier_score(block["home_win"], block["market_prob_home"]),
        }
        if spread_col is not None:
            row["model_mae"] = mae(block["margin"], block[spread_col])
            row["market_mae"] = mae(block["margin"], block["market_spread"])
        return row

    rows = {season: one_block(block) for season, block in scored.groupby("season")}
    rows["all"] = one_block(scored)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "season"
    return table


def calibration_plot(
    y: np.ndarray | pd.Series,
    p: np.ndarray | pd.Series,
    path: Path,
    title: str = "Calibration",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Plot predicted probability vs. actual win rate in bins and save it.

    A calibrated model's points sit on the 45-degree line: of games given a 70%
    chance, about 70% are won.

    Args:
        y: 0/1 outcomes.
        p: Predicted probabilities.
        path: Where to save the PNG.
        title: Plot title.
        n_bins: Number of equal-width probability bins.

    Returns:
        The binned table (mean predicted, actual rate, games per bin).
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = (
        df.groupby("bin", observed=True)
        .agg(predicted=("p", "mean"), actual=("y", "mean"), games=("y", "size"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, label="Perfect calibration")
    ax.scatter(table["predicted"], table["actual"], s=table["games"] / table["games"].max() * 200)
    ax.plot(table["predicted"], table["actual"], marker="", linewidth=1)
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return table
