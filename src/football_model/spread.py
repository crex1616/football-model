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
    """

    # Defaults are the best stage-1 settings from scripts/tune_spread.py (MAE on
    # 2015-2024 excl. 2020). A 16-week recency half-life or a 42-point margin cap
    # improved MAE by only 0.01 or less, within noise, so both are left off.
    alpha: float = 3.0
    revert: float = 0.1
    margin_cap: float | None = None
    half_life_weeks: float | None = None


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


def fit_ratings(
    train: pd.DataFrame,
    prior: pd.Series,
    params: SpreadParams,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """Fit ratings on `train` by ridge regression, shrinking toward `prior`.

    Solves b = m + (X'WX + alpha I)^-1 X'W (y - X m), where m is the prior. Home field
    is shrunk toward its prior too, so a week-2 refit on a handful of games can't swing
    it far from last season's value.

    Args:
        train: Games to fit on (home_team, away_team, neutral_site, margin, start_date).
        prior: Prior mean for each coefficient. Index: "home_field" then every team.
        params: Model settings (alpha, margin_cap, half_life_weeks).
        as_of: Reference time for recency weights (the next kickoff). Needed only when
            params.half_life_weeks is set.

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
    m = prior.to_numpy()
    residual = y.to_numpy() - x @ m
    xtw = x.T * w
    lhs = xtw @ x + params.alpha * np.eye(x.shape[1])
    deviation = np.linalg.solve(lhs, xtw @ residual)
    return pd.Series(m + deviation, index=prior.index)


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
