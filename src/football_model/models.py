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

Phase 5c adds two models alongside the stack, on the same features and splits:

| Model | Target | Econometrics analogue |
|---|---|---|
| stack | margin | OLS with a ridge penalty, then Phi(spread / sigma) |
| logistic | home_win | L2-penalized logit; C = 1 / lambda (small C = more shrinkage) |
| LightGBM | margin | flexible nonparametric regression: trees fit one at a time, each to the
  residuals of the trees before it; finds interactions and nonlinearities on its own |

The trees fit training data much better than they predict new games, so LightGBM's
sigma is estimated out of sample (holdout_sigma), not from training residuals.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
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


# --- Phase 5c: penalized logistic regression ---


def fit_logistic(train: pd.DataFrame, cols: list[str], C: float) -> Pipeline:
    """Fit an L2-penalized logit of home_win on standardized features.

    The statistical idea: logit(P(home win)) = b0 + sum_k b_k * z_k, fitted by maximum
    likelihood minus a penalty lambda * sum_k b_k^2 (the intercept is not penalized).
    sklearn writes the penalty as C = 1 / lambda, so a smaller C shrinks more.

    Unlike the stack, this model targets the win itself, not the margin: it never sees
    how big a win was, but it needs no normality assumption or sigma.

    Args:
        train: Rows with home_win and every column in cols.
        cols: Feature columns.
        C: Inverse penalty strength.

    Returns:
        Fitted scaler + logistic regression.
    """
    model = Pipeline(
        [("scale", StandardScaler()), ("logit", LogisticRegression(C=C, max_iter=1000))]
    )
    model.fit(train[cols], train["home_win"].astype(int))
    return model


def walk_forward_logistic(
    features: pd.DataFrame,
    cols: list[str],
    C: float,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Predict each season from a logistic model trained on earlier seasons only.

    Returns:
        New DataFrame with game_id and logit_prob (P(home win)) for games in `seasons`.
    """
    parts = []
    for season in seasons:
        model = fit_logistic(training_rows(features, season), cols, C)
        test = features[features["season"] == season]
        parts.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "logit_prob": model.predict_proba(test[cols])[:, 1],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def logit_coefficient_table(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Unpenalized logit of home_win on standardized features, for reading.

    Each coefficient is the change in log-odds of a home win per one-SD change in the
    feature, holding the others fixed. Dividing by about 1.7 puts it on the probit
    scale, comparable to the stack's coefficient / sigma.

    Returns:
        DataFrame indexed by feature: coef, std_err, z, p_value.
    """
    z = (train[cols] - train[cols].mean()) / train[cols].std()
    result = sm.Logit(train["home_win"].astype(int), sm.add_constant(z)).fit(disp=0)
    return pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "z": result.tvalues,
            "p_value": result.pvalues,
        }
    )


# --- Phase 5c: LightGBM ---


@dataclass(frozen=True)
class LgbmParams:
    """LightGBM settings that control how flexible the trees are.

    - num_leaves: leaves per tree; more leaves = deeper interactions.
    - learning_rate: how much of each tree's fit is kept (shrinkage per step).
    - n_estimators: number of trees. With learning_rate, this sets total fit; too many
      and the model starts fitting noise, like adding too many regressors.
    - min_child_samples: smallest number of games allowed in a leaf (a floor on the
      sample size behind each fitted value).
    """

    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int


def fit_lgbm(train: pd.DataFrame, cols: list[str], params: LgbmParams) -> LGBMRegressor:
    """Fit gradient-boosted trees to the home margin (squared-error loss).

    Trees don't need standardized inputs: a split on a feature is the same whatever its
    units. Seeded and single-threaded so results are reproducible.
    """
    model = LGBMRegressor(
        **asdict(params),
        random_state=config.RANDOM_SEED,
        deterministic=True,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(train[cols], train["margin"])
    return model


def holdout_sigma(
    features: pd.DataFrame, season: int, cols: list[str], params: LgbmParams
) -> float:
    """Out-of-sample residual SD for LightGBM, from the last training season.

    Takes the training rows for `season`, holds out the last of those seasons, fits on
    the seasons before it, and returns the SD of the held-out residuals. This keeps time
    order (the held-out season comes after the seasons it was predicted from) and never
    touches `season` itself. In-sample residuals would understate sigma for trees and
    make every probability overconfident.

    When the held-out season is the only training season (2015, when predicting 2016),
    the fit falls back to the burn-in season. Its features are weak (ratings start from
    scratch), so that sigma comes out slightly large: conservative, as in
    win_prob.walk_forward_sigma.

    Args:
        features: Feature table with season, margin, and every column in cols.
        season: The season being forecast.
        cols: Feature columns.
        params: LightGBM settings.

    Returns:
        SD of (margin - prediction) on the last training season, in points.
    """
    train = training_rows(features, season)
    last = train["season"].max()
    fit_rows = training_rows(features, last)
    if fit_rows.empty:
        fit_rows = features[features["season"] == config.BURN_IN_SEASON]
    model = fit_lgbm(fit_rows, cols, params)
    held_out = train[train["season"] == last]
    residuals = held_out["margin"] - model.predict(held_out[cols])
    return float(residuals.std())


def walk_forward_lgbm(
    features: pd.DataFrame,
    cols: list[str],
    params: LgbmParams,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Predict each season from LightGBM trained on earlier seasons only.

    Sigma comes from holdout_sigma, so each season's probabilities use only games
    before that season.

    Returns:
        New DataFrame with game_id, lgbm_spread (expected home margin), lgbm_prob
        (P(home win) = Phi(spread / sigma)), and lgbm_sigma, for games in `seasons`.
    """
    parts = []
    for season in seasons:
        sigma = holdout_sigma(features, season, cols, params)
        model = fit_lgbm(training_rows(features, season), cols, params)
        test = features[features["season"] == season]
        pred = model.predict(test[cols])
        parts.append(
            pd.DataFrame(
                {
                    "game_id": test["game_id"].to_numpy(),
                    "lgbm_spread": pred,
                    "lgbm_prob": norm.cdf(pred / sigma),
                    "lgbm_sigma": np.full(len(test), sigma),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def lgbm_importance(model: LGBMRegressor, cols: list[str]) -> pd.Series:
    """Share of the total loss reduction from splits on each feature ("gain" importance).

    Roughly: how much each feature helped the trees fit. It has no sign and no standard
    error, so it is a guide to what the model leans on, not a coefficient.
    """
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(gain / gain.sum(), index=cols).sort_values(ascending=False)
