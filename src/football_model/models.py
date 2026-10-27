"""Models that combine pre-game features into a spread and a win probability.

Phase 5b has one: the stacked linear model. It is a ridge regression of the home
margin on standardized features (from features.py), turned into a probability with
Phi(spread / sigma) like the Phase 4 model:

    margin_g = b0 + sum_k b_k * z_{g,k} + e_g,     z = (x - mean) / sd on training rows

"Stacked" because some inputs are themselves model outputs (the ridge, Huber, and Elo
ratings): this regression learns how much weight each deserves, alongside the new
features. Standardizing first puts every coefficient in "points per standard deviation"
so the ridge penalty treats them equally; the mean and SD come from training seasons
only, so the test season never influences the scaling.

Training is walk-forward by season: to predict season N, fit on every season before N.
The burn-in season (ratings start from scratch) and the 2020 COVID season (short,
irregular schedules) are left out of training; 2020 still feeds the features
themselves, e.g. as the prior for 2021 ratings.
"""

from collections.abc import Iterable

import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_model import config

EXCLUDED_TRAINING_SEASONS = (config.BURN_IN_SEASON, config.COVID_SEASON)


def training_rows(features: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rows used to train the model that predicts `season`: earlier seasons, minus exclusions."""
    earlier = features["season"] < season
    allowed = ~features["season"].isin(EXCLUDED_TRAINING_SEASONS)
    return features[earlier & allowed]


def fit_stacked(train: pd.DataFrame, cols: list[str], alpha: float) -> tuple[Pipeline, float]:
    """Fit the stacked ridge model on training rows.

    Args:
        train: Rows with margin and every column in cols.
        cols: Feature columns.
        alpha: Ridge penalty on the standardized coefficients.

    Returns:
        (pipeline, sigma): the fitted scaler + ridge, and the SD of training residuals
        (used as sigma in Phi(spread / sigma); in-sample, so very slightly small, but
        with thousands of games and a handful of features the difference is tiny).
    """
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(train[cols], train["margin"])
    residuals = train["margin"] - model.predict(train[cols])
    return model, float(residuals.std())


def walk_forward_stacked(
    features: pd.DataFrame,
    cols: list[str],
    alpha: float,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Predict each season from a stacked model trained on earlier seasons only.

    Args:
        features: features.build_features output.
        cols: Feature columns.
        alpha: Ridge penalty.
        seasons: Seasons to predict.

    Returns:
        New DataFrame with game_id, stack_spread (expected home margin), stack_prob
        (P(home win)), for the games in `seasons`.
    """
    parts = []
    for season in seasons:
        model, sigma = fit_stacked(training_rows(features, season), cols, alpha)
        test = features[features["season"] == season]
        pred = model.predict(test[cols])
        parts.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "stack_spread": pred,
                    "stack_prob": norm.cdf(pred / sigma),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def coefficient_table(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """OLS of margin on standardized features, with standard errors and t-stats.

    This is the unpenalized version of the stacked model, for reading: each coefficient
    is points of margin per one-SD change in the feature, holding the others fixed.
    Expect large standard errors on features that overlap (the three ratings measure
    nearly the same thing); that hurts inference, not prediction.

    Returns:
        DataFrame indexed by feature: coef, std_err, t, p_value.
    """
    z = (train[cols] - train[cols].mean()) / train[cols].std()
    result = sm.OLS(train["margin"], sm.add_constant(z)).fit()
    return pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "t": result.tvalues,
            "p_value": result.pvalues,
        }
    )


# The Phase 5b stack, as selected by scripts/evaluate_features.py: all four feature
# groups, ridge penalty 100. It runs in "shadow" in the weekly predictions (recorded
# next to the live model, not used as it) because it beat the baseline on the tuning
# seasons but not on the 2025 test season (docs/phase5-plan.md).
STACK_ALPHA = 100.0
STACK_FEATURES = [
    "home",
    "ridge_spread",
    "huber_spread",
    "elo_diff",
    "epa_edge",
    "sr_edge",
    "ret_ppa_diff",
    "ret_ppa_diff_faded",
    "rest_diff",
    "travel_diff",
]


def predict_stack(training: pd.DataFrame, upcoming: pd.DataFrame, season: int) -> pd.DataFrame:
    """Spread and win probability from the stack for games in `season`.

    Args:
        training: Feature table with past seasons (features.load_features()).
        upcoming: Rows with every column in STACK_FEATURES.
        season: The season being predicted; the model trains on seasons before it.

    Returns:
        New DataFrame aligned with `upcoming`: stack_spread, stack_home_win_prob.
    """
    model, sigma = fit_stacked(training_rows(training, season), STACK_FEATURES, STACK_ALPHA)
    spread_pred = model.predict(upcoming[STACK_FEATURES])
    return pd.DataFrame(
        {"stack_spread": spread_pred, "stack_home_win_prob": norm.cdf(spread_pred / sigma)},
        index=upcoming.index,
    )
