"""Turn a predicted spread into a home win probability.

The statistical idea: if the actual margin is the predicted spread plus normal noise,

    margin = spread + e,   e ~ N(0, sigma^2)

then P(home win) = P(margin > 0) = Phi(spread / sigma). This is a probit model with no
intercept and the slope fixed at 1 / sigma, where sigma is the standard deviation of
the model's out-of-sample residuals. Nothing is fitted except sigma.

Three versions live here, kept side by side:

- normal: one sigma per season, estimated from every *earlier* season's residuals
  (walk-forward, so a season's probabilities never use its own results).
- normal_by_week: a separate sigma for each part of the season (weeks 1-4, 5-8, 9+,
  postseason). Early-season ratings know less, so their residuals are bigger; one
  pooled sigma would make September probabilities overconfident. In econometrics terms,
  this models heteroskedasticity in the error term.
- logistic: logit(P) = a + b * spread, fitted by maximum likelihood on earlier seasons.
  This is the cross-check. It frees the intercept and slope: a should be near 0 (home
  field is already in the spread), and b should be near 1.7 / sigma (logit coefficients
  are about 1.7 times probit coefficients). A slope well away from that means the
  spreads are on the wrong scale, i.e. systematically over- or under-confident.

Results (scripts/evaluate_win_prob.py, ridge spreads, tuning seasons 2015-2024 excl.
2020): normal 0.4999 log loss, logistic 0.5000, normal_by_week 0.5003; Elo 0.5166,
market 0.4736. normal is the default: simplest and best. The logistic fit agrees with
it (slope 0.1006 vs. 1.7 / sigma = 0.1009; intercept 0.05), so the spreads are on the
right scale. Separate sigmas by week did not help: early-season residuals are bigger
mostly because of FCS mismatches, where the win probability is near 1 either way.

Ties are not possible in college football (overtime), so home_win = margin > 0.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from football_model import evaluate


def residual_sd(
    games: pd.DataFrame,
    spread_col: str,
    seasons: Iterable[int],
    by: pd.Series | None = None,
) -> float | pd.Series:
    """Standard deviation of (margin - spread) on the given seasons.

    Args:
        games: Must contain season, margin, and spread_col.
        spread_col: Column with the predicted home margin.
        seasons: Seasons to use.
        by: Optional labels aligned with `games` (e.g. evaluate.week_bucket(games));
            if given, return one SD per label.

    Returns:
        A single SD in points, or a Series of SDs indexed by label.
    """
    subset = games[games["season"].isin(list(seasons))]
    resid = subset["margin"] - subset[spread_col]
    if by is None:
        return float(resid.std())
    return resid.groupby(by.loc[subset.index]).std()


def walk_forward_sigma(games: pd.DataFrame, spread_col: str, by_week: bool = False) -> pd.Series:
    """Per-game sigma estimated from every earlier season's residuals only.

    The first season has no earlier seasons, so its sigma is NaN (it is the burn-in
    season and is never scored). The burn-in season's own residuals are used for later
    seasons; they are larger than usual (ratings start from scratch), which makes the
    second season's sigma slightly conservative.

    Args:
        games: Must contain season, margin, spread_col, and (with by_week) week and
            season_type.
        spread_col: Column with the predicted home margin.
        by_week: Estimate one sigma per week bucket instead of one per season.

    Returns:
        Series aligned with `games`.
    """
    sigma = pd.Series(np.nan, index=games.index)
    buckets = evaluate.week_bucket(games) if by_week else None
    for season in sorted(games["season"].unique()):
        earlier = [s for s in games["season"].unique() if s < season]
        if not earlier:
            continue
        this = games["season"] == season
        if by_week:
            sds = residual_sd(games, spread_col, earlier, by=buckets)
            sigma[this] = buckets[this].map(sds)
        else:
            sigma[this] = residual_sd(games, spread_col, earlier)
    return sigma


def normal_prob(games: pd.DataFrame, spread_col: str, by_week: bool = False) -> pd.Series:
    """P(home win) = Phi(spread / sigma), with sigma from walk_forward_sigma.

    Args:
        games: See walk_forward_sigma.
        spread_col: Column with the predicted home margin.
        by_week: Use a separate sigma per week bucket.

    Returns:
        Series of probabilities aligned with `games` (NaN for the first season).
    """
    sigma = walk_forward_sigma(games, spread_col, by_week)
    probs = evaluate.market_win_prob(games[spread_col], sigma.to_numpy())
    return pd.Series(probs, index=games.index)


def fit_logistic(games: pd.DataFrame, spread_col: str, seasons: Iterable[int]):
    """Logistic regression of home_win on the predicted spread, on the given seasons.

    Args:
        games: Must contain season, home_win, and spread_col.
        spread_col: Column with the predicted home margin.
        seasons: Seasons to fit on.

    Returns:
        The fitted statsmodels Logit result. Call .summary() for coefficients and
        standard errors; .params holds [const, spread_col].
    """
    subset = games[games["season"].isin(list(seasons))]
    X = sm.add_constant(subset[[spread_col]])
    return sm.Logit(subset["home_win"], X).fit(disp=0)


def logistic_prob(games: pd.DataFrame, spread_col: str) -> pd.Series:
    """Walk-forward logistic probabilities: each season uses a fit on earlier seasons only.

    Args:
        games: Must contain season, home_win, and spread_col.
        spread_col: Column with the predicted home margin.

    Returns:
        Series of probabilities aligned with `games` (NaN for the first season).
    """
    probs = pd.Series(np.nan, index=games.index)
    for season in sorted(games["season"].unique()):
        earlier = [s for s in games["season"].unique() if s < season]
        if not earlier:
            continue
        result = fit_logistic(games, spread_col, earlier)
        this = games["season"] == season
        X = sm.add_constant(games.loc[this, [spread_col]], has_constant="add")
        probs[this] = result.predict(X)
    return probs


def add_win_probs(games: pd.DataFrame, spread_col: str, prefix: str) -> pd.DataFrame:
    """Add all three win-probability versions for one spread model.

    Args:
        games: Must contain season, week, season_type, margin, home_win, spread_col.
        spread_col: Column with the predicted home margin.
        prefix: Name for the new columns, e.g. "ridge".

    Returns:
        New DataFrame with {prefix}_prob_normal, {prefix}_prob_normal_by_week, and
        {prefix}_prob_logistic.
    """
    games = games.copy()
    games[f"{prefix}_prob_normal"] = normal_prob(games, spread_col)
    games[f"{prefix}_prob_normal_by_week"] = normal_prob(games, spread_col, by_week=True)
    games[f"{prefix}_prob_logistic"] = logistic_prob(games, spread_col)
    return games


def current_sigma(pred: pd.DataFrame, spread_col: str, season: int) -> float:
    """Sigma for live predictions in `season`: residual SD on every earlier season.

    The same estimate walk_forward_sigma gives a season it hasn't seen.

    Args:
        pred: Walk-forward predictions (run_spread output) with margin and spread_col.
        spread_col: Column with the predicted home margin.
        season: The season being predicted.

    Returns:
        Sigma in points.
    """
    earlier = [s for s in pred["season"].unique() if s < season]
    return residual_sd(pred, spread_col, earlier)
