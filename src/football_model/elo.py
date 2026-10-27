"""Elo ratings for college football.

Every team has one number, its rating. Before a game, the rating gap (plus home
field) gives the home team's expected win probability. After the game, both
ratings move toward the result:

    E_home     = 1 / (1 + 10^(-(R_home - R_away + H * home) / 400))
    R_home_new = R_home + K * M * (S_home - E_home)
    R_away_new = R_away - K * M * (S_home - E_home)

S_home is 1 for a home win and 0 for a loss, `home` is 0 at neutral sites, and M is
an optional margin-of-victory multiplier.

The statistical idea: Elo is an online, recursive estimator of team strength, much
like exponential smoothing or a simple Kalman filter. K is the step size: how much
weight the newest game gets relative to everything before it. A big K reacts fast
but is noisy (high variance); a small K is stable but slow to notice real change
(high bias). Because each prediction uses only games already played, every Elo
prediction is out-of-sample by construction. The overfitting risk is in *choosing*
K and the other settings, which is why they are tuned on validation seasons and
checked once on a held-out test season.
"""

import math
from dataclasses import dataclass

import pandas as pd

FCS_TEAM = "FCS"


@dataclass(frozen=True)
class EloParams:
    """Settings for an Elo run. These are hyperparameters: chosen by validation, not fitted.

    Attributes:
        k: Step size for each update (rating points per unit of surprise).
        home_field: Home-field advantage in Elo points (zeroed at neutral sites).
        revert: Share of each team's distance from the FBS mean removed between seasons
            (0 = ratings carry over fully, 1 = everyone starts the season equal).
        use_mov: Scale updates by the margin-of-victory multiplier.
        initial: Starting rating for every team in the first season.
    """

    # Defaults are the best settings from scripts/tune_elo.py (log loss on 2015-2024
    # excl. 2020). The log loss surface is flat near the optimum, so nearby values
    # (k 30-40, home_field 40-60) score within 0.001 of these.
    k: float = 35.0
    home_field: float = 50.0
    revert: float = 0.2
    use_mov: bool = True
    initial: float = 1500.0


def expected_home_win(
    r_home: float, r_away: float, home_field: float, neutral: bool = False
) -> float:
    """Pre-game probability that the home team wins.

    This is a logistic curve in the rating gap: a 400-point edge means 10-to-1 odds.

    Args:
        r_home: Home team rating before the game.
        r_away: Away team rating before the game.
        home_field: Home-field advantage in Elo points.
        neutral: True for a neutral-site game, which removes home field.

    Returns:
        Probability between 0 and 1.
    """
    diff = r_home - r_away + (0.0 if neutral else home_field)
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    """Margin-of-victory multiplier (FiveThirtyEight's form).

    M = ln(|margin| + 1) * 2.2 / (0.001 * winner_elo_diff + 2.2)

    The log term makes a 28-point win count for more than a 3-point win, with
    diminishing returns for running up the score. The second term corrects for
    autocorrelation: favorites tend to win by more, so without it, strong teams'
    ratings would keep inflating. It shrinks the update when the favorite wins big
    and enlarges it when an underdog does.

    Args:
        margin: Final margin (sign ignored).
        winner_elo_diff: Winner's pre-game rating minus loser's, including home field.

    Returns:
        A positive multiplier, around 1-3 for typical games.
    """
    return math.log(abs(margin) + 1.0) * 2.2 / (0.001 * winner_elo_diff + 2.2)


def _revert_to_mean(ratings: dict[str, float], revert: float) -> dict[str, float]:
    """Pull every FBS rating part of the way back to the FBS mean between seasons.

    This is shrinkage toward the mean: last year's rating is a noisy guess about this
    year's team (graduation, transfers, coaching changes). The shared FCS team is a
    pool of many schools rather than one roster, so it is not reverted.
    """
    fbs = {team: r for team, r in ratings.items() if team != FCS_TEAM}
    if not fbs:
        return dict(ratings)
    mean = sum(fbs.values()) / len(fbs)
    reverted = {team: mean + (1.0 - revert) * (r - mean) for team, r in fbs.items()}
    if FCS_TEAM in ratings:
        reverted[FCS_TEAM] = ratings[FCS_TEAM]
    return reverted


def run_elo(games: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """Run Elo through every game in time order and record each game's pre-game ratings.

    Ratings are written down *before* each game updates them, so the prediction for
    a game never uses that game or any later one (no leakage).

    Teams seen for the first time after the first season (programs moving up to FBS)
    start at the current FCS rating, since they were FCS teams the year before.

    Args:
        games: Games table from clean.py (needs season, start_date, home_team,
            away_team, neutral_site, margin). Sorted here by start_date.
        params: Elo settings.

    Returns:
        New DataFrame: the games plus home_elo_pre, away_elo_pre, elo_diff (home minus
        away including home field), elo_prob_home, home_elo_post, away_elo_post.
    """
    games = games.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    first_season = games["season"].min()

    ratings: dict[str, float] = {}
    current_season = first_season
    records = []

    for game in games.itertuples(index=False):
        if game.season != current_season:
            ratings = _revert_to_mean(ratings, params.revert)
            current_season = game.season

        for team in (game.home_team, game.away_team):
            if team not in ratings:
                is_new_program = game.season != first_season and team != FCS_TEAM
                if is_new_program and FCS_TEAM in ratings:
                    ratings[team] = ratings[FCS_TEAM]
                else:
                    ratings[team] = params.initial

        r_home = ratings[game.home_team]
        r_away = ratings[game.away_team]
        home_edge = 0.0 if game.neutral_site else params.home_field
        diff = r_home - r_away + home_edge
        prob_home = expected_home_win(r_home, r_away, params.home_field, game.neutral_site)

        result = 1.0 if game.margin > 0 else 0.0
        multiplier = 1.0
        if params.use_mov:
            winner_diff = diff if game.margin > 0 else -diff
            multiplier = mov_multiplier(game.margin, winner_diff)

        shift = params.k * multiplier * (result - prob_home)
        ratings[game.home_team] = r_home + shift
        ratings[game.away_team] = r_away - shift

        records.append(
            {
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
                "elo_diff": diff,
                "elo_prob_home": prob_home,
                "home_elo_post": r_home + shift,
                "away_elo_post": r_away - shift,
            }
        )

    return pd.concat([games, pd.DataFrame(records)], axis=1)


def fit_points_per_elo(rated_games: pd.DataFrame, seasons: list[int]) -> float:
    """Estimate how many Elo points equal one point of margin, on the given seasons.

    OLS through the origin of margin on elo_diff: margin = b * elo_diff + e. The
    slope b is points of margin per Elo point (about 1/25). Fit it on training
    seasons only, then use elo_spread = b * elo_diff on later seasons.

    Args:
        rated_games: Output of run_elo.
        seasons: Seasons to fit on.

    Returns:
        The slope b.
    """
    subset = rated_games[rated_games["season"].isin(seasons)]
    x = subset["elo_diff"]
    y = subset["margin"]
    return float((x * y).sum() / (x * x).sum())


def predict(games: pd.DataFrame, params: EloParams, fit_seasons: list[int]) -> pd.DataFrame:
    """Run Elo and add a predicted spread in points.

    elo_spread = b * elo_diff, where b comes from fit_points_per_elo on fit_seasons.

    Args:
        games: Games table from clean.py.
        params: Elo settings.
        fit_seasons: Seasons used to estimate points per Elo point (training only).

    Returns:
        Output of run_elo plus elo_spread (expected home margin).
    """
    rated = run_elo(games, params)
    rated["elo_spread"] = fit_points_per_elo(rated, fit_seasons) * rated["elo_diff"]
    return rated


def final_ratings(rated_games: pd.DataFrame) -> pd.DataFrame:
    """Each team's rating after its most recent game, highest first.

    Args:
        rated_games: Output of run_elo.

    Returns:
        DataFrame with columns team, rating, last_game (start_date), sorted by rating.
    """
    home = rated_games[["home_team", "home_elo_post", "start_date"]].set_axis(
        ["team", "rating", "last_game"], axis=1
    )
    away = rated_games[["away_team", "away_elo_post", "start_date"]].set_axis(
        ["team", "rating", "last_game"], axis=1
    )
    latest = (
        pd.concat([home, away])
        .sort_values("last_game")
        .drop_duplicates("team", keep="last")
        .sort_values("rating", ascending=False)
    )
    return latest.reset_index(drop=True)


def upcoming_elo_diff(games: pd.DataFrame, upcoming: pd.DataFrame, params: EloParams) -> pd.Series:
    """Pre-game Elo difference (home minus away, plus home field) for unplayed games.

    Runs Elo through every completed game before the first upcoming kickoff and uses
    each team's latest rating. If the upcoming games are in a new season, ratings are
    reverted toward the mean first, as run_elo does at a season change. Teams never
    rated get the FCS rating (as new programs do in run_elo).

    Args:
        games: Completed games (clean_games output).
        upcoming: Games to predict, one season (clean_schedule output).
        params: Elo settings.

    Returns:
        Series aligned with `upcoming`: elo_diff, the same quantity run_elo records.
    """
    history = games[games["start_date"] < upcoming["start_date"].min()]
    rated = run_elo(history, params)
    latest = final_ratings(rated)
    ratings = dict(zip(latest["team"], latest["rating"]))
    if int(upcoming["season"].iloc[0]) > int(history["season"].max()):
        ratings = _revert_to_mean(ratings, params.revert)

    fallback = ratings.get(FCS_TEAM, params.initial)
    r_home = upcoming["home_team"].map(ratings).fillna(fallback)
    r_away = upcoming["away_team"].map(ratings).fillna(fallback)
    home_edge = (~upcoming["neutral_site"].astype(bool)).astype(float) * params.home_field
    return (r_home - r_away + home_edge).astype(float)
