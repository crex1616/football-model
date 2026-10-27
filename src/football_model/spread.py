"""Point-spread ratings from a ridge regression on game margins.

Every game is one row of a regression:

    margin_g = h * home_g + sum_t beta_t * x_{g,t} + e_g

x_{g,t} is +1 if team t is home in game g, -1 if it is away, and 0 otherwise; home_g is
1 unless the game is at a neutral site. Each beta is a team's rating in points versus an
average team, h is home-field advantage in points, and the predicted spread (expected
home margin) is h * home + beta_home - beta_away.

The statistical idea: this is a two-way fixed-effects model with a ridge penalty.
Ridge is the posterior mean of a Bayesian regression with a Normal prior on each
coefficient, centered on a prior mean m with variance tau^2:

    b = argmin  sum_g w_g (y_g - x_g b)^2  +  alpha * ||b - m||^2
      = m + (X'WX + alpha I)^-1 X'W (y - X m)

alpha = sigma_e^2 / tau^2: the larger it is, the more the estimates are pulled toward
the prior mean. Here the prior mean is last season's rating (shrunk toward average), so
early in a season, when each team has played two or three games, ratings stay close to
last year's; as games accumulate, the data takes over.

Identification: every row of the team columns sums to zero (+1 and -1), so adding the
same constant to every beta leaves every prediction unchanged. OLS has no unique
answer; the penalty picks one. Only rating *differences* mean anything.

Unlike Elo, which updates once per game and never looks back, this refits on every game
played so far each week, so strength of schedule flows both ways: if a team you beat in
week 2 turns out to be strong, the week-8 refit raises the credit for that win.

Two models live here:

- The ridge model, SpreadParams(): squared loss, so a result 40 points better than
  expected pulls the ratings twice as hard as one 20 points better.
- The Huber model, HUBER_PARAMS: Huber loss, the robust-regression (M-estimator) fix for
  blowouts and garbage-time scoring. Residuals up to c points are squared as usual;
  beyond c the loss grows only linearly, so any result more than c points better than
  expected counts as "c points better" toward the ratings. This caps the *surprise*,
  not the margin: a 40-point favorite still gets a 40-point predicted spread. (Clipping
  the margin itself, margin_cap, was tested in Phase 3 and hurt badly, because it also
  caps every prediction for a mismatch.)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FCS_TEAM = "FCS"
HOME_FIELD = "home_field"


@dataclass(frozen=True)
class SpreadParams:
    """Settings for the ridge spread model. Hyperparameters: chosen by validation, not fitted.

    Attributes:
        alpha: Ridge penalty. In Bayesian terms sigma_e^2 / tau^2: residual variance over
            prior variance of a team's deviation from its preseason rating.
        revert: Share of each team's distance from the FBS average removed between
            seasons (0 = last season's rating carries over fully, 1 = everyone starts equal).
        margin_cap: If set, margins are clipped to [-cap, cap] in the training target, so
            a 63-0 blowout counts like a cap-point win. Predictions are still scored against
            true margins.
        half_life_weeks: If set, a game played this many weeks before the prediction gets
            half the weight of a game played this week (weighted least squares). None
            weights every game equally.
        huber_c: If set, use Huber loss with threshold c points instead of squared loss
            (see huber_weights). None is the plain ridge model.
    """

    # Defaults are the best stage-1 settings from scripts/tune_spread.py (MAE on
    # 2015-2024 excl. 2020). A 16-week recency half-life or a 42-point margin cap
    # improved MAE by only 0.01 or less, within noise, so both are left off.
    alpha: float = 3.0
    revert: float = 0.1
    margin_cap: float | None = None
    half_life_weeks: float | None = None
    huber_c: float | None = None


# The Huber model: the same ridge regression with Huber loss. Settings are the best from
# scripts/tune_spread_huber.py (MAE on 2015-2024 excl. 2020). SpreadParams() remains
# the plain ridge model.
# c = 21 beat 14 and 28 but tied ridge: tuning MAE 13.350 vs. 13.356.
HUBER_PARAMS = SpreadParams(alpha=3.0, revert=0.1, huber_c=21.0)


def design_matrix(games: pd.DataFrame, teams: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Build the regression's X and y for a set of games.

    Args:
        games: Games with home_team, away_team, neutral_site, margin.
        teams: Team columns, in order. Every team in `games` must be listed.

    Returns:
        X: one row per game; a home_field column (1, or 0 at neutral sites) then one
            column per team (+1 home, -1 away, 0 otherwise).
        y: margin (home points minus away points).
    """
    position = {team: i for i, team in enumerate(teams)}
    rows = np.arange(len(games))
    x = np.zeros((len(games), len(teams)))
    x[rows, games["home_team"].map(position).to_numpy()] = 1.0
    x[rows, games["away_team"].map(position).to_numpy()] = -1.0

    X = pd.DataFrame(x, columns=teams, index=games.index)
    X.insert(0, HOME_FIELD, (~games["neutral_site"]).astype(float).to_numpy())
    y = games["margin"].astype(float)
    return X, y


def recency_weights(
    start_dates: pd.Series, as_of: pd.Timestamp, half_life_weeks: float | None
) -> np.ndarray:
    """Weight each game by 0.5 ** (weeks before as_of / half_life_weeks); all 1 if None."""
    if half_life_weeks is None:
        return np.ones(len(start_dates))
    weeks_ago = (as_of - start_dates).dt.total_seconds().to_numpy() / (7 * 24 * 3600)
    return 0.5 ** (weeks_ago / half_life_weeks)


def huber_weights(residuals: np.ndarray, c: float) -> np.ndarray:
    """IRLS weights for Huber loss: 1 where |residual| <= c, else c / |residual|.

    Weighted least squares with these weights has the same first-order conditions as
    Huber loss, so a game with residual 2c gets weight 1/2 and pulls the fit as hard
    as a residual of c would. Games inside c are untouched.
    """
    abs_r = np.abs(residuals)
    return np.where(abs_r <= c, 1.0, c / np.maximum(abs_r, 1e-12))


def ridge_solve(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, m: np.ndarray, alpha: float
) -> np.ndarray:
    """b = m + (X'WX + alpha I)^-1 X'W (y - X m)."""
    xtw = x.T * w
    lhs = xtw @ x + alpha * np.eye(x.shape[1])
    return m + np.linalg.solve(lhs, xtw @ (y - x @ m))


def fit_ratings(
    train: pd.DataFrame,
    prior: pd.Series,
    params: SpreadParams,
    as_of: pd.Timestamp | None = None,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> pd.Series:
    """Fit ratings on `train` by ridge regression, shrinking toward `prior`.

    Solves b = m + (X'WX + alpha I)^-1 X'W (y - X m), where m is the prior. Home field
    is shrunk toward its prior too, so a week-2 refit on a handful of games can't swing
    it far from last season's value.

    With params.huber_c set, the loss is Huber instead of squared, solved by iteratively
    reweighted least squares (IRLS, as in statsmodels' RLM): fit, compute residuals,
    downweight games with |residual| > c (huber_weights), refit, and repeat until the
    coefficients stop moving.

    Args:
        train: Games to fit on (home_team, away_team, neutral_site, margin, start_date).
        prior: Prior mean for each coefficient. Index: "home_field" then every team.
        params: Model settings (alpha, margin_cap, half_life_weeks, huber_c).
        as_of: Reference time for recency weights (the next kickoff). Needed only when
            params.half_life_weeks is set.
        max_iter: Maximum IRLS iterations (Huber only).
        tol: Stop IRLS once no coefficient moves more than this many points.

    Returns:
        Fitted coefficients, same index as `prior`: home field and team ratings in points.
    """
    teams = list(prior.index.drop(HOME_FIELD))
    X, y = design_matrix(train, teams)
    if params.margin_cap is not None:
        y = y.clip(-params.margin_cap, params.margin_cap)

    if params.half_life_weeks is not None and as_of is None:
        raise ValueError("as_of is required when half_life_weeks is set")
    w = recency_weights(train["start_date"], as_of, params.half_life_weeks)

    x = X.to_numpy()
    y_arr = y.to_numpy()
    m = prior.to_numpy()
    b = ridge_solve(x, y_arr, w, m, params.alpha)

    if params.huber_c is not None:
        for _ in range(max_iter):
            robust = huber_weights(y_arr - x @ b, params.huber_c)
            b_new = ridge_solve(x, y_arr, w * robust, m, params.alpha)
            converged = np.max(np.abs(b_new - b)) < tol
            b = b_new
            if converged:
                break

    return pd.Series(b, index=prior.index)


def season_prior(last_fit: pd.Series | None, teams: list[str], revert: float) -> pd.Series:
    """Preseason prior for each coefficient, from last season's final fit.

    Ratings are re-centered so the average FBS team is 0, then each FBS team keeps
    (1 - revert) of its distance from average: shrinkage toward the mean, as in Elo,
    because rosters turn over. The pooled FCS team is not reverted. Programs new to FBS
    start at the FCS rating (they were FCS last year). Home field carries over unchanged.

    Args:
        last_fit: Output of fit_ratings for last season's full set of games, or None for
            the first season (every prior is then 0).
        teams: Teams playing this season.
        revert: Share of distance from average removed.

    Returns:
        Series indexed by "home_field" then `teams`.
    """
    index = [HOME_FIELD, *teams]
    if last_fit is None:
        return pd.Series(0.0, index=index)

    old = last_fit.drop(HOME_FIELD)
    fbs_mean = old.drop(FCS_TEAM, errors="ignore").mean()
    centered = old - fbs_mean
    fcs_rating = centered.get(FCS_TEAM, 0.0)

    prior = {HOME_FIELD: float(last_fit[HOME_FIELD])}
    for team in teams:
        if team == FCS_TEAM:
            prior[team] = fcs_rating
        elif team in centered.index:
            prior[team] = (1.0 - revert) * centered[team]
        else:
            prior[team] = fcs_rating
    return pd.Series(prior)[index]


def _season_teams(season_games: pd.DataFrame) -> list[str]:
    """Sorted list of every team appearing in the games."""
    return sorted(set(season_games["home_team"]) | set(season_games["away_team"]))


def _walk_forward(
    games: pd.DataFrame, params: SpreadParams
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Run the weekly refits; return the predictions and each season's final fit."""
    games = games.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    columns = ["home_rating_pre", "away_rating_pre", "hfa_pre", "spread_pred", "games_in_fit"]
    out = pd.DataFrame(np.nan, index=games.index, columns=columns)

    final_fits: dict[int, pd.Series] = {}
    last_fit = None
    for season, season_games in games.groupby("season", sort=True):
        teams = _season_teams(season_games)
        prior = season_prior(last_fit, teams, params.revert)

        # A block is one (season_type, week). It is predicted from a fit on every game
        # of this season that kicked off before the block's first game: no leakage.
        blocks = season_games.groupby(["season_type", "week"])["start_date"].min()
        for key, first_kickoff in blocks.sort_values().items():
            train = season_games[season_games["start_date"] < first_kickoff]
            fit = prior if train.empty else fit_ratings(train, prior, params, first_kickoff)

            in_block = (season_games["season_type"] == key[0]) & (season_games["week"] == key[1])
            block = season_games[in_block]
            home_rating = block["home_team"].map(fit).to_numpy()
            away_rating = block["away_team"].map(fit).to_numpy()
            home_edge = np.where(block["neutral_site"], 0.0, fit[HOME_FIELD])
            out.loc[block.index, "home_rating_pre"] = home_rating
            out.loc[block.index, "away_rating_pre"] = away_rating
            out.loc[block.index, "hfa_pre"] = home_edge
            out.loc[block.index, "spread_pred"] = home_edge + home_rating - away_rating
            out.loc[block.index, "games_in_fit"] = len(train)

        as_of = season_games["start_date"].max()
        last_fit = fit_ratings(season_games, prior, params, as_of)
        final_fits[int(season)] = last_fit

    out["games_in_fit"] = out["games_in_fit"].astype(int)
    return pd.concat([games, out], axis=1), final_fits


def run_spread(games: pd.DataFrame, params: SpreadParams) -> pd.DataFrame:
    """Walk-forward spread predictions for every game.

    Within a season, each week is predicted from a ridge fit on that season's earlier
    games, shrunk toward the preseason prior. At season's end, a fit on all its games
    becomes next season's prior (after reverting toward average). The first season has
    a zero prior, so treat it as burn-in and don't score it.

    Args:
        games: Games table from clean.py (needs game_id, season, week, season_type,
            start_date, home_team, away_team, neutral_site, margin).
        params: Model settings.

    Returns:
        New DataFrame sorted by start_date: the games plus home_rating_pre,
        away_rating_pre, hfa_pre (0 at neutral sites), spread_pred (expected home
        margin, + = home favored), and games_in_fit (games the ratings were fit on).
    """
    predictions, _ = _walk_forward(games, params)
    return predictions


def final_ratings(games: pd.DataFrame, params: SpreadParams, season: int) -> pd.DataFrame:
    """End-of-season ratings for one season, best first, centered on the average FBS team.

    Args:
        games: Games table from clean.py, including every season up to `season`
            (earlier seasons supply the prior).
        params: Model settings.
        season: Season to report.

    Returns:
        DataFrame with team, rating (points vs. average FBS team), and games played.
        The home-field estimate is in the frame's attrs["home_field"].
    """
    _, fits = _walk_forward(games[games["season"] <= season], params)
    fit = fits[season]
    ratings = fit.drop(HOME_FIELD)
    ratings = ratings - ratings.drop(FCS_TEAM, errors="ignore").mean()

    season_games = games[games["season"] == season]
    played = pd.concat([season_games["home_team"], season_games["away_team"]]).value_counts()
    table = pd.DataFrame(
        {
            "team": ratings.index,
            "rating": ratings.to_numpy(),
            "games": played[ratings.index].to_numpy(),
        }
    )
    table = table.sort_values("rating", ascending=False).reset_index(drop=True)
    table.attrs["home_field"] = float(fit[HOME_FIELD])
    return table


def predict_upcoming(
    games: pd.DataFrame, upcoming: pd.DataFrame, params: SpreadParams
) -> pd.DataFrame:
    """Predict spreads for games not yet played, from a fit on every game before them.

    This is the live version of one step of run_spread: the season's ratings are fit on
    all of its games that kicked off before the first upcoming game, shrunk toward the
    preseason prior from last season. Games on or after that kickoff are dropped first,
    so nothing from the future can enter the fit.

    Args:
        games: Completed games (clean_games output), last season and earlier plus the
            current season so far.
        upcoming: Games to predict (clean_schedule output), all in the current season.
        params: Model settings.

    Returns:
        New DataFrame: `upcoming` plus home_rating, away_rating, hfa (0 at neutral
        sites), spread_pred (expected home margin, + = home favored), and games_in_fit.
    """
    first_kickoff = upcoming["start_date"].min()
    season = int(upcoming["season"].iloc[0])
    history = games[games["start_date"] < first_kickoff]

    _, fits = _walk_forward(history, params)
    if season in fits:
        fit = fits[season]
        games_in_fit = int((history["season"] == season).sum())
    else:
        # No games played yet this season: the preseason prior is the prediction.
        teams = sorted(set(upcoming["home_team"]) | set(upcoming["away_team"]))
        fit = season_prior(fits.get(season - 1), teams, params.revert)
        games_in_fit = 0

    # A team with no rating (e.g. new to FBS, no games yet) gets the FCS rating, as in
    # season_prior.
    fallback = float(fit.get(FCS_TEAM, 0.0))
    out = upcoming.copy()
    out["home_rating"] = out["home_team"].map(fit).fillna(fallback).to_numpy()
    out["away_rating"] = out["away_team"].map(fit).fillna(fallback).to_numpy()
    out["hfa"] = np.where(out["neutral_site"], 0.0, fit[HOME_FIELD])
    out["spread_pred"] = out["hfa"] + out["home_rating"] - out["away_rating"]
    out["games_in_fit"] = games_in_fit
    return out
