"""Pre-game features for every game: one row per game, home team's perspective.

Every feature is something knowable before kickoff, and every one is a home-minus-away
difference. They come in groups (FEATURE_GROUPS) so a model can add them one group at
a time and keep a group only if it improves walk-forward log loss:

- ratings: the existing walk-forward ratings (ridge and Huber spreads, Elo).
- efficiency: opponent-adjusted offense and defense, in EPA per play and success rate.
- preseason: returning production, faded out as games are played.
- situational: rest days and travel distance.

No recruiting data is used (user decision): neither recruiting class rankings nor the
talent composite, which is built from players' recruiting ratings.

How leakage is avoided: ratings are fit only on games that kicked off before the game
being described (the same block-by-block walk-forward as spread.py); counts and rest
days look only at a team's earlier games; returning production is published before
the season.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_model import config, data, elo, spread

FCS_TEAM = "FCS"
HOME_FIELD = "home_field"
CONSTANT = "const"
REST_CAP_DAYS = 14

FEATURE_GROUPS = {
    "ratings": ["home", "ridge_spread", "huber_spread", "elo_diff"],
    "efficiency": ["epa_edge", "sr_edge"],
    "preseason": ["ret_ppa_diff", "ret_ppa_diff_faded"],
    "situational": ["rest_diff", "travel_diff"],
}

# Advanced-stats columns used for the efficiency ratings (garbage time excluded by CFBD).
METRICS = {"epa": "offense.ppa", "sr": "offense.successRate"}


@dataclass(frozen=True)
class EfficiencyParams:
    """Settings for the opponent-adjusted efficiency ratings. Chosen by validation.

    Attributes:
        alpha: Ridge penalty: residual variance over prior variance of a team's
            offense or defense rating (same idea as SpreadParams.alpha).
        revert: Share of each team's distance from average removed between seasons.
    """

    alpha: float = 10.0
    revert: float = 0.3


# --- Efficiency ratings -------------------------------------------------------------


def game_efficiency(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each game's offensive EPA per play and success rate for both teams.

    Advanced stats come one row per team-game with CFBD school names. Teams that the
    games table pools as "FCS" are matched by elimination: in a game, the row whose
    team is not the FBS side belongs to the FCS side.

    Args:
        stats: Advanced game stats (data.load_advanced_game_stats), several seasons.
        games: Games table (clean.py) with game_id, home_team, away_team.

    Returns:
        New DataFrame with game_id and home_/away_ versions of each metric in METRICS
        (e.g. home_epa = EPA per play of the home offense). NaN where CFBD has no stats.
    """
    keep = ["gameId", "team", *METRICS.values()]
    rows = stats[keep].rename(columns={"gameId": "game_id"})
    rows = rows.merge(games[["game_id", "home_team", "away_team"]], on="game_id")

    is_home = (rows["team"] == rows["home_team"]) | (
        (rows["home_team"] == FCS_TEAM) & (rows["team"] != rows["away_team"])
    )
    out = games[["game_id"]].copy()
    for side, mask in [("home", is_home), ("away", ~is_home)]:
        side_rows = rows.loc[mask, ["game_id", *METRICS.values()]]
        side_rows = side_rows.drop_duplicates("game_id")
        side_rows = side_rows.rename(columns={col: f"{side}_{m}" for m, col in METRICS.items()})
        out = out.merge(side_rows, on="game_id", how="left")
    return out


def offense_rows(games: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Reshape games to two rows per game, one per offense, for the efficiency regression.

    Args:
        games: Games with home_team, away_team, neutral_site, start_date, and
            home_{metric} / away_{metric} (from game_efficiency).
        metric: A key of METRICS, e.g. "epa".

    Returns:
        New DataFrame: offense, defense, home (+1 offense at home, -1 offense away, 0 at
        a neutral site), y (the offense's value), start_date. Games without stats are
        dropped.
    """
    venue = np.where(games["neutral_site"], 0.0, 1.0)
    home_side = pd.DataFrame(
        {
            "offense": games["home_team"],
            "defense": games["away_team"],
            "home": venue,
            "y": games[f"home_{metric}"],
            "start_date": games["start_date"],
        }
    )
    away_side = pd.DataFrame(
        {
            "offense": games["away_team"],
            "defense": games["home_team"],
            "home": -venue,
            "y": games[f"away_{metric}"],
            "start_date": games["start_date"],
        }
    )
    rows = pd.concat([home_side, away_side], ignore_index=True)
    return rows.dropna(subset=["y"]).reset_index(drop=True)


def efficiency_design_matrix(
    rows: pd.DataFrame, teams: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """X and y for the offense/defense regression.

        y = c + h * home + O_offense - D_defense + e

    O is offensive strength and D is defensive strength, both "higher is better", so a
    good defense lowers the opponent's expected value.

    Args:
        rows: offense_rows output.
        teams: Every team in rows, in column order.

    Returns:
        X with columns const, home_field, O:<team>..., D:<team>...; and y.
    """
    position = {team: i for i, team in enumerate(teams)}
    n = len(rows)
    idx = np.arange(n)
    off = np.zeros((n, len(teams)))
    off[idx, rows["offense"].map(position).to_numpy()] = 1.0
    dfn = np.zeros((n, len(teams)))
    dfn[idx, rows["defense"].map(position).to_numpy()] = -1.0

    columns = [f"O:{t}" for t in teams] + [f"D:{t}" for t in teams]
    X = pd.DataFrame(np.hstack([off, dfn]), columns=columns, index=rows.index)
    X.insert(0, HOME_FIELD, rows["home"].to_numpy())
    X.insert(0, CONSTANT, 1.0)
    return X, rows["y"].astype(float)


def efficiency_prior(last_fit: pd.Series | None, teams: list[str], revert: float) -> pd.Series:
    """Preseason prior for the efficiency coefficients, from last season's final fit.

    Offense and defense ratings are each re-centered on the FBS average and pulled
    (revert) toward it, like spread.season_prior. New programs start at the FCS values.
    The constant and home field carry over unchanged.
    """
    index = [CONSTANT, HOME_FIELD] + [f"O:{t}" for t in teams] + [f"D:{t}" for t in teams]
    if last_fit is None:
        return pd.Series(0.0, index=index)

    prior = {CONSTANT: float(last_fit[CONSTANT]), HOME_FIELD: float(last_fit[HOME_FIELD])}
    for side in ["O", "D"]:
        old = last_fit[[k for k in last_fit.index if k.startswith(f"{side}:")]]
        old.index = [k[2:] for k in old.index]
        centered = old - old.drop(FCS_TEAM, errors="ignore").mean()
        fcs_value = centered.get(FCS_TEAM, 0.0)
        for team in teams:
            if team != FCS_TEAM and team in centered.index:
                prior[f"{side}:{team}"] = (1.0 - revert) * centered[team]
            else:
                prior[f"{side}:{team}"] = fcs_value
    return pd.Series(prior)[index]


def fit_efficiency(
    rows: pd.DataFrame, teams: list[str], prior: pd.Series, alpha: float
) -> pd.Series:
    """Ridge fit of the offense/defense model, shrunk toward `prior`.

    b = m + (X'X + alpha I)^-1 X'(y - X m), the same estimator as spread.fit_ratings.

    Args:
        rows: offense_rows output (training games only).
        teams: Teams with coefficients (every team in rows).
        prior: Prior mean for each coefficient (efficiency_prior).
        alpha: Ridge penalty.

    Returns:
        Coefficients with the same index as `prior` (the prior itself if rows is empty).
    """
    if rows.empty:
        return prior
    X, y = efficiency_design_matrix(rows, teams)
    b = spread.ridge_solve(X.to_numpy(), y.to_numpy(), np.ones(len(y)), prior.to_numpy(), alpha)
    return pd.Series(b, index=prior.index)


def run_efficiency(games: pd.DataFrame, metric: str, params: EfficiencyParams) -> pd.DataFrame:
    """Walk-forward pre-game offense and defense ratings for every game.

    Within a season, each block of games (one season_type and week) gets ratings from a
    ridge fit on that season's games that kicked off before the block's first game,
    shrunk toward the preseason prior (the same estimator and block structure as
    spread.py). At season's end, a fit on all its games becomes next season's prior.

    Args:
        games: Games table plus home_{metric} / away_{metric} (game_efficiency).
        metric: A key of METRICS.
        params: Penalty and between-season reversion.

    Returns:
        New DataFrame aligned with `games` (same index): {metric}_home_off,
        {metric}_home_def, {metric}_away_off, {metric}_away_def, {metric}_const,
        {metric}_hfa, all as they stood before kickoff.
    """
    ordered = games.sort_values(["start_date", "game_id"])
    names = ["home_off", "home_def", "away_off", "away_def", "const", "hfa"]
    out = pd.DataFrame(np.nan, index=games.index, columns=[f"{metric}_{n}" for n in names])

    last_fit = None
    for _, season_games in ordered.groupby("season", sort=True):
        teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
        prior = efficiency_prior(last_fit, teams, params.revert)
        rows = offense_rows(season_games, metric)

        # A block is one (season_type, week), predicted from a fit on this season's
        # games that kicked off before the block's first game: no leakage.
        blocks = season_games.groupby(["season_type", "week"])["start_date"].min()
        for key, first_kickoff in blocks.sort_values().items():
            train = rows[rows["start_date"] < first_kickoff]
            fit = fit_efficiency(train, teams, prior, params.alpha)
            offense = {t: fit[f"O:{t}"] for t in teams}
            defense = {t: fit[f"D:{t}"] for t in teams}

            in_block = (season_games["season_type"] == key[0]) & (season_games["week"] == key[1])
            block = season_games[in_block]
            out.loc[block.index, f"{metric}_home_off"] = block["home_team"].map(offense)
            out.loc[block.index, f"{metric}_home_def"] = block["home_team"].map(defense)
            out.loc[block.index, f"{metric}_away_off"] = block["away_team"].map(offense)
            out.loc[block.index, f"{metric}_away_def"] = block["away_team"].map(defense)
            out.loc[block.index, f"{metric}_const"] = fit[CONSTANT]
            out.loc[block.index, f"{metric}_hfa"] = fit[HOME_FIELD]

        last_fit = fit_efficiency(rows, teams, prior, params.alpha)
    return out


def efficiency_edge(ratings: pd.DataFrame, metric: str) -> pd.Series:
    """(O_home + D_home) - (O_away + D_away): the home team's net efficiency edge.

    The expected home-offense value minus the expected away-offense value is
    2h + this edge (at a non-neutral site); home field is left to the model's other
    features.
    """
    home = ratings[f"{metric}_home_off"] + ratings[f"{metric}_home_def"]
    away = ratings[f"{metric}_away_off"] + ratings[f"{metric}_away_def"]
    return home - away


def predicted_offense_value(
    games: pd.DataFrame, ratings: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """Pre-game predicted vs. actual offensive value for each side (for tuning alpha).

    Returns:
        New DataFrame with predicted and actual columns, two rows per game with stats.
    """
    venue = np.where(games["neutral_site"], 0.0, 1.0)
    c, h = ratings[f"{metric}_const"], ratings[f"{metric}_hfa"]
    home_pred = c + h * venue + ratings[f"{metric}_home_off"] - ratings[f"{metric}_away_def"]
    away_pred = c - h * venue + ratings[f"{metric}_away_off"] - ratings[f"{metric}_home_def"]
    out = pd.DataFrame(
        {
            "season": pd.concat([games["season"], games["season"]], ignore_index=True),
            "predicted": pd.concat([home_pred, away_pred], ignore_index=True),
            "actual": pd.concat(
                [games[f"home_{metric}"], games[f"away_{metric}"]], ignore_index=True
            ),
        }
    )
    return out.dropna()


# --- Counts, rest, travel -----------------------------------------------------------


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team per game (team identified by CFBD id)."""
    cols = ["game_id", "season", "start_date"]
    home = games[cols].assign(side="home", team_id=games["home_id"].to_numpy())
    away = games[cols].assign(side="away", team_id=games["away_id"].to_numpy())
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["start_date", "game_id", "side"]).reset_index(drop=True)


def _to_game_columns(games: pd.DataFrame, long: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a per-team value back to home_{value} and away_{value}, aligned with games."""
    wide = long.pivot(index="game_id", columns="side", values=value)
    out = pd.DataFrame(index=games.index)
    out[f"home_{value}"] = games["game_id"].map(wide["home"]).to_numpy()
    out[f"away_{value}"] = games["game_id"].map(wide["away"]).to_numpy()
    return out


def games_played_before(games: pd.DataFrame) -> pd.DataFrame:
    """How many games each team had played this season before this game.

    Counts only earlier rows of the same team and season (cumcount), so the game itself
    and later games never enter.

    Returns:
        New DataFrame aligned with games: home_played, away_played.
    """
    long = _team_games(games)
    long["played"] = long.groupby(["season", "team_id"]).cumcount()
    return _to_game_columns(games, long, "played")


def rest_days(games: pd.DataFrame, cap: int = REST_CAP_DAYS) -> pd.DataFrame:
    """Days since each team's previous game this season, capped at `cap`.

    A team's first game of the season (and any longer break) counts as `cap` days:
    beyond about two weeks, extra rest is assumed not to matter.

    Returns:
        New DataFrame aligned with games: home_rest, away_rest.
    """
    long = _team_games(games)
    previous = long.groupby(["season", "team_id"])["start_date"].shift(1)
    days = (long["start_date"] - previous).dt.total_seconds() / 86400.0
    long["rest"] = days.fillna(cap).clip(upper=cap)
    return _to_game_columns(games, long, "rest")


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles between two sets of coordinates (degrees)."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def travel_miles(games: pd.DataFrame, teams: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Distance from each team's home stadium to the game's venue, in miles.

    Args:
        games: Games with home_id, away_id, venue_id.
        teams: data.load_teams output (id, location.latitude, location.longitude).
        venues: data.load_venues output (id, latitude, longitude).

    Returns:
        New DataFrame aligned with games: home_travel, away_travel (NaN if a location
        is missing).
    """
    team_loc = teams.set_index("id")[["location.latitude", "location.longitude"]]
    venue_loc = venues.set_index("id")[["latitude", "longitude"]]
    v_lat = games["venue_id"].map(venue_loc["latitude"])
    v_lon = games["venue_id"].map(venue_loc["longitude"])
    out = pd.DataFrame(index=games.index)
    for side in ["home", "away"]:
        t_lat = games[f"{side}_id"].map(team_loc["location.latitude"])
        t_lon = games[f"{side}_id"].map(team_loc["location.longitude"])
        out[f"{side}_travel"] = haversine_miles(t_lat, t_lon, v_lat, v_lon)
    return out


# --- Preseason ----------------------------------------------------------------------


def returning_ppa(games: pd.DataFrame, returning: pd.DataFrame) -> pd.DataFrame:
    """Share of last season's EPA each team brings back (published before the season).

    Teams without a value (pooled FCS, first-year FBS programs, teams back from a pause)
    get that season's average, so their difference carries no information.

    percentPPA is returning EPA divided by last season's total EPA, which explodes when
    a weak team's total is near zero (UMass 2021: -3.1). It is clipped to [0, 1]: none
    to all of last season's production is back.

    Args:
        games: Games with season, home_team, away_team.
        returning: data.load_returning_production output, several seasons.

    Returns:
        New DataFrame aligned with games: home_ret_ppa, away_ret_ppa.
    """
    returning = returning.drop_duplicates(["season", "team"])
    returning = returning.assign(percentPPA=returning["percentPPA"].clip(0.0, 1.0))
    table = returning.set_index(["season", "team"])["percentPPA"]
    season_mean = returning.groupby("season")["percentPPA"].mean()
    out = pd.DataFrame(index=games.index)
    for side in ["home", "away"]:
        keys = pd.MultiIndex.from_arrays([games["season"], games[f"{side}_team"]])
        values = pd.Series(table.reindex(keys).to_numpy(), index=games.index)
        out[f"{side}_ret_ppa"] = values.fillna(games["season"].map(season_mean))
    return out


# --- Everything together ------------------------------------------------------------


def build_features(
    games: pd.DataFrame,
    efficiency_params: dict[str, EfficiencyParams] | None = None,
) -> pd.DataFrame:
    """All feature groups for every game in the games table.

    Loads the cached Phase 5 data (no API calls once cached). Missing situational
    values (a venue or team with no coordinates) become a 0 difference.

    Args:
        games: Games table from clean.py (completed seasons).
        efficiency_params: EfficiencyParams per metric key; defaults if None.

    Returns:
        New DataFrame sorted by start_date: the games plus every column in
        FEATURE_GROUPS and the intermediate pieces (pre-game ratings, rest, travel,
        games played) for inspection.
    """
    efficiency_params = efficiency_params or {m: EfficiencyParams() for m in METRICS}
    games = games.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(games["season"].unique())

    # Group 1: existing walk-forward ratings.
    ridge = spread.run_spread(games, spread.SpreadParams())
    huber = spread.run_spread(games, spread.HUBER_PARAMS)
    rated = elo.run_elo(games, elo.EloParams())
    out = games.copy()
    out["home"] = (~out["neutral_site"]).astype(float)
    out["ridge_spread"] = out["game_id"].map(ridge.set_index("game_id")["spread_pred"])
    out["huber_spread"] = out["game_id"].map(huber.set_index("game_id")["spread_pred"])
    out["elo_diff"] = out["game_id"].map(rated.set_index("game_id")["elo_diff"])

    # Group 2: opponent-adjusted efficiency.
    stats = pd.concat([data.load_advanced_game_stats(s) for s in seasons], ignore_index=True)
    out = out.merge(game_efficiency(stats, out), on="game_id", how="left")
    for metric, params in efficiency_params.items():
        ratings = run_efficiency(out, metric, params)
        out = pd.concat([out, ratings], axis=1)
        out[f"{metric}_edge"] = efficiency_edge(ratings, metric)

    # Group 3: preseason strength, fading as games are played.
    returning = pd.concat([data.load_returning_production(s) for s in seasons], ignore_index=True)
    out = pd.concat([out, returning_ppa(out, returning), games_played_before(out)], axis=1)
    out["ret_ppa_diff"] = out["home_ret_ppa"] - out["away_ret_ppa"]
    avg_played = (out["home_played"] + out["away_played"]) / 2
    out["ret_ppa_diff_faded"] = out["ret_ppa_diff"] / (1 + avg_played)

    # Group 4: situational.
    out = pd.concat([out, rest_days(out)], axis=1)
    # A pooled FCS team's games against other FCS teams aren't in the table, so its
    # rest looks like the full cap every time; no rest edge is computed for those games.
    has_fcs = (out["home_team"] == FCS_TEAM) | (out["away_team"] == FCS_TEAM)
    out["rest_diff"] = (out["home_rest"] - out["away_rest"]).where(~has_fcs, 0.0)
    travel = travel_miles(out, data.load_teams(), data.load_venues())
    out = pd.concat([out, travel], axis=1)
    out["travel_diff"] = ((out["away_travel"] - out["home_travel"]) / 100).fillna(0.0)
    return out


def save_features(features: pd.DataFrame) -> None:
    """Write the feature table to config.FEATURES_TABLE_PATH."""
    config.FEATURES_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURES_TABLE_PATH, index=False)


def load_features() -> pd.DataFrame:
    """Read the saved feature table (scripts/evaluate_features.py builds it)."""
    return pd.read_parquet(config.FEATURES_TABLE_PATH)


def tune_efficiency(
    games: pd.DataFrame,
    metric: str,
    alphas: list[float],
    reverts: list[float],
    seasons: list[int],
) -> pd.DataFrame:
    """Mean squared error of pre-game predicted offensive value, for each setting.

    Picks the efficiency ratings' alpha and revert the same way the spread model was
    tuned: by how well the walk-forward ratings predict games they haven't seen.

    Args:
        games: Games table merged with game_efficiency output.
        metric: A key of METRICS.
        alphas: Ridge penalties to try.
        reverts: Between-season reversion shares to try.
        seasons: Seasons to score (the tuning seasons).

    Returns:
        DataFrame with alpha, revert, mse, sorted best first.
    """
    rows = []
    for alpha in alphas:
        for revert in reverts:
            ratings = run_efficiency(games, metric, EfficiencyParams(alpha, revert))
            scored = predicted_offense_value(games, ratings, metric)
            scored = scored[scored["season"].isin(seasons)]
            mse = float(((scored["actual"] - scored["predicted"]) ** 2).mean())
            rows.append({"alpha": alpha, "revert": revert, "mse": mse})
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
