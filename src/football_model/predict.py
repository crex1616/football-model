"""Live weekly predictions: build them and save them to predictions/YYYY/ before kickoff.

A saved prediction file is the honest track record, so it is never overwritten or
edited once written (CLAUDE.md rule 4). save_predictions enforces that: it refuses to
write over an existing file, and refuses to include a game that has already kicked off.

Models in each file, kept side by side:

- ridge: the Phase 3 ridge spread with Phi(spread / sigma) (the primary model).
- huber: the Huber-loss spread with Phi(spread / sigma).

market_spread is the consensus line when the file was written. It is saved so the
week can later be scored "model vs. market"; it is never a model input.
"""

from pathlib import Path

import pandas as pd

from football_model import config, evaluate, spread, win_prob

MODELS = {"ridge": spread.SpreadParams(), "huber": spread.HUBER_PARAMS}

OUTPUT_COLUMNS = [
    "created_at",
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "home_team",
    "away_team",
    "neutral_site",
    "home_rating",
    "away_rating",
    "hfa",
    "ridge_spread",
    "ridge_home_win_prob",
    "huber_spread",
    "huber_home_win_prob",
    "sigma",
    "games_in_fit",
    "market_spread",
]


def prediction_path(season: int, week: int) -> Path:
    """predictions/{season}/week_{WW}.csv."""
    return config.PREDICTIONS_DIR / str(season) / f"week_{week:02d}.csv"


def build_week_predictions(
    history: pd.DataFrame, upcoming: pd.DataFrame, created_at: pd.Timestamp
) -> pd.DataFrame:
    """Spreads and win probabilities from every model in MODELS for the upcoming games.

    For each model: ratings from spread.predict_upcoming, sigma from that model's
    walk-forward residuals on all earlier seasons (win_prob.current_sigma), and
    P(home win) = Phi(spread / sigma). Ratings, home field, and sigma columns are the
    ridge model's.

    Args:
        history: Completed games, earlier seasons plus this season so far.
        upcoming: Games to predict (clean_schedule output with market_spread).
        created_at: Timestamp written to every row (UTC).

    Returns:
        New DataFrame with OUTPUT_COLUMNS, sorted by kickoff.
    """
    season = int(upcoming["season"].iloc[0])
    out = upcoming.copy()
    for name, params in MODELS.items():
        pred = spread.predict_upcoming(history, upcoming, params)
        backtest = spread.run_spread(history[history["season"] < season], params)
        sigma = win_prob.current_sigma(backtest, "spread_pred", season)

        out[f"{name}_spread"] = pred["spread_pred"].to_numpy()
        out[f"{name}_home_win_prob"] = evaluate.market_win_prob(pred["spread_pred"], sigma)
        if name == "ridge":
            for col in ["home_rating", "away_rating", "hfa", "games_in_fit"]:
                out[col] = pred[col].to_numpy()
            out["sigma"] = sigma

    out["created_at"] = created_at
    out = out.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    return out[OUTPUT_COLUMNS]


def drop_started(predictions: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Keep only games that kick off after `now`, adding a note that the week is partial."""
    remaining = predictions[predictions["start_date"] > now].copy()
    n_dropped = len(predictions) - len(remaining)
    remaining["note"] = f"partial week: {n_dropped} game(s) had already kicked off"
    return remaining


def save_predictions(predictions: pd.DataFrame, path: Path, now: pd.Timestamp) -> None:
    """Write a week's predictions, refusing anything that would break the track record.

    Args:
        predictions: build_week_predictions output (optionally after drop_started).
        path: Destination (prediction_path).
        now: Current time (UTC).

    Raises:
        FileExistsError: If `path` already exists. Corrections go in a new file.
        ValueError: If any game has already kicked off.
    """
    if path.exists():
        raise FileExistsError(f"{path} already exists; prediction files are never overwritten.")
    started = predictions["start_date"] <= now
    if started.any():
        raise ValueError(
            f"{int(started.sum())} game(s) have already kicked off; "
            "use --allow-partial to save only the games not yet started."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def format_line(spread_value: float, home_team: str, away_team: str, exact: bool = False) -> str:
    """Write an expected home margin the way a sportsbook prints it: the favorite, minus.

    Our convention is + = home favored, so +7.0 becomes "Home -7.0" and -6.5 becomes
    "Away -6.5". A spread within 0.05 of zero is "Pick". Missing is "-".

    exact=True keeps up to two decimals instead of rounding to one, for market lines:
    the consensus averages several books, so it can land on a quarter point (5.75).
    """
    if pd.isna(spread_value):
        return "-"
    if abs(spread_value) < 0.05:
        return "Pick"
    favorite = home_team if spread_value > 0 else away_team
    points = f"{abs(spread_value):.2f}" if exact else f"{abs(spread_value):.1f}"
    if exact and points.endswith("0"):
        points = points[:-1]
    return f"{favorite} -{points}"


def readable_table(
    predictions: pd.DataFrame, timezone: str = config.DISPLAY_TIMEZONE
) -> pd.DataFrame:
    """A table for reading in the terminal, in kickoff order.

    Columns: the ridge model's line and the market line in sportsbook style (favorite
    minus), the ridge favorite's win probability, kickoff in local time, and the
    matchup ("Away @ Home", or "vs" at a neutral site).

    Args:
        predictions: build_week_predictions output.
        timezone: Time zone for kickoff times.

    Returns:
        New DataFrame of strings, one row per game.
    """
    preds = predictions.sort_values(["start_date", "game_id"])
    rows = []
    for game in preds.itertuples():
        p_home = game.ridge_home_win_prob
        favorite = game.home_team if p_home >= 0.5 else game.away_team
        separator = "vs" if game.neutral_site else "@"
        rows.append(
            {
                "Predicted line": format_line(game.ridge_spread, game.home_team, game.away_team),
                "Market line": format_line(
                    game.market_spread, game.home_team, game.away_team, exact=True
                ),
                "Win %": f"{favorite} {max(p_home, 1 - p_home):.0%}",
                "Kickoff": game.start_date.tz_convert(timezone).strftime("%a %b %d %I:%M %p"),
                "Matchup": f"{game.away_team} {separator} {game.home_team}",
            }
        )
    return pd.DataFrame(rows)
