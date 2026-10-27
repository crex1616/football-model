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
    "home_conference",
    "away_conference",
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


def excel_path(season: int, week: int) -> Path:
    """predictions/{season}/week_{WW}.xlsx (a readable copy of the CSV)."""
    return prediction_path(season, week).with_suffix(".xlsx")


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


def load_saved_predictions(path: Path) -> pd.DataFrame:
    """Read a saved prediction CSV, with kickoff and creation times parsed as UTC."""
    preds = pd.read_csv(path)
    for col in ["start_date", "created_at"]:
        preds[col] = pd.to_datetime(preds[col], utc=True)
    return preds


def add_conferences(predictions: pd.DataFrame, raw_games: pd.DataFrame) -> pd.DataFrame:
    """Add home_conference and away_conference from raw CFBD games, matched on game_id.

    Only for prediction files saved before conferences were part of OUTPUT_COLUMNS
    (week 9 of 2026). Conferences are known before kickoff, so this changes nothing
    about the predictions themselves.
    """
    lookup = raw_games[["id", "homeConference", "awayConference"]].rename(
        columns={
            "id": "game_id",
            "homeConference": "home_conference",
            "awayConference": "away_conference",
        }
    )
    return predictions.merge(lookup, on="game_id", how="left", validate="one_to_one")


# Excel columns: the easy-to-read columns first, then the model's raw numbers.
EXCEL_PERCENT_COLUMNS = ["Win %", "Home win % (ridge)", "Home win % (Huber)"]
EXCEL_NUMBER_COLUMNS = [
    "Home rating",
    "Away rating",
    "Home field (pts)",
    "Ridge spread (home pts)",
    "Huber spread (home pts)",
    "Market spread (home pts)",
    "Sigma",
]


def excel_table(predictions: pd.DataFrame, timezone: str = config.DISPLAY_TIMEZONE) -> pd.DataFrame:
    """One row per game for the Excel file, in kickoff order.

    First the teams, then the predicted and market lines in sportsbook style (favorite
    minus), the model favorite's win probability, and kickoff in local time. The raw
    model numbers (ratings, home field, spreads as expected home margin, sigma) are at
    the end.

    Args:
        predictions: build_week_predictions output (or a saved CSV), with conferences.
        timezone: Time zone for kickoff times.

    Returns:
        New DataFrame with display column names.
    """
    preds = predictions.sort_values(["start_date", "game_id"])
    rows = []
    for game in preds.itertuples():
        p_home = game.ridge_home_win_prob
        rows.append(
            {
                "Away": game.away_team,
                "Home": game.home_team,
                "Predicted line": format_line(game.ridge_spread, game.home_team, game.away_team),
                "Market line": format_line(
                    game.market_spread, game.home_team, game.away_team, exact=True
                ),
                "Model favorite": game.home_team if p_home >= 0.5 else game.away_team,
                "Win %": max(p_home, 1 - p_home),
                "Kickoff": game.start_date.tz_convert(timezone).strftime("%a %b %d %I:%M %p"),
                "Site": "Neutral" if game.neutral_site else "Home",
                "Away conference": game.away_conference,
                "Home conference": game.home_conference,
                "Huber line": format_line(game.huber_spread, game.home_team, game.away_team),
                "Home win % (ridge)": p_home,
                "Home win % (Huber)": game.huber_home_win_prob,
                "Home rating": game.home_rating,
                "Away rating": game.away_rating,
                "Home field (pts)": game.hfa,
                "Ridge spread (home pts)": game.ridge_spread,
                "Huber spread (home pts)": game.huber_spread,
                "Market spread (home pts)": game.market_spread,
                "Sigma": game.sigma,
                "Games in fit": game.games_in_fit,
                "Game ID": game.game_id,
                "Created (UTC)": game.created_at.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return pd.DataFrame(rows)


def conference_tabs(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split an excel_table into tabs: "All games", then one per FBS conference.

    A game between two conferences appears on both tabs. Pooled FCS opponents don't get
    a tab of their own; the game shows on the FBS team's tab.
    """
    tabs = {"All games": table}
    conferences = set()
    for side in ["Away", "Home"]:
        is_fbs = table[side] != spread.FCS_TEAM
        conferences |= set(table.loc[is_fbs, f"{side} conference"].dropna())
    for conference in sorted(conferences):
        home_in = (table["Home conference"] == conference) & (table["Home"] != spread.FCS_TEAM)
        away_in = (table["Away conference"] == conference) & (table["Away"] != spread.FCS_TEAM)
        # Excel sheet names: at most 31 characters, no / \ ? * [ ] :
        name = "".join(ch for ch in conference if ch not in "/\\?*[]:")[:31]
        tabs[name] = table[home_in | away_in].reset_index(drop=True)
    return tabs


def write_excel(predictions: pd.DataFrame, path: Path) -> None:
    """Write the week's predictions as an Excel workbook, one tab per conference.

    The workbook is a readable copy of the CSV, which stays the permanent record, so it
    can be rebuilt from the CSV at any time (scripts/export_predictions.py).

    Args:
        predictions: build_week_predictions output (or a saved CSV), with conferences.
        path: Destination .xlsx file.
    """
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    tabs = conference_tabs(excel_table(predictions))
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tabs.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "C2"  # keep the header row and team names in view
            for i, column in enumerate(table.columns, start=1):
                letter = get_column_letter(i)
                sheet[f"{letter}1"].font = Font(bold=True)
                values = [str(v) for v in table[column]] + [column]
                sheet.column_dimensions[letter].width = min(max(map(len, values)) + 2, 40)
                if column in EXCEL_PERCENT_COLUMNS or column in EXCEL_NUMBER_COLUMNS:
                    number_format = "0%" if column in EXCEL_PERCENT_COLUMNS else "0.0"
                    for cell in sheet[letter][1:]:
                        cell.number_format = number_format


def results_path(season: int, week: int) -> Path:
    """predictions/{season}/week_{WW}_results.csv: a week's predictions joined to results."""
    return config.PREDICTIONS_DIR / str(season) / f"week_{week:02d}_results.csv"


def ats_result(model_spread: float, closing_spread: float, margin: float) -> str:
    """Did the side the model liked against the closing line cover? "win", "loss", or "push".

    All three numbers are expected or actual home margin. The model likes the home team
    against the line when model_spread > closing_spread; home covers when margin >
    closing_spread. Equal model and closing lines (no side) or a margin exactly on the
    line is a push. For reading only: against-the-spread records are noisy and are not
    how the model is judged (that is log loss and MAE).
    """
    if model_spread == closing_spread or margin == closing_spread:
        return "push"
    likes_home = model_spread > closing_spread
    home_covered = margin > closing_spread
    return "win" if likes_home == home_covered else "loss"


def score_predictions(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    closing: pd.DataFrame,
    market_sigma: float,
) -> pd.DataFrame:
    """Join saved predictions to final results and the closing line, one row per finished game.

    Args:
        predictions: A saved prediction file (load_saved_predictions).
        results: Finished games (clean.clean_games output) with margin and home_win.
            Games not in here (not yet played) are left out.
        closing: Closing consensus lines (clean.consensus_lines output).
        market_sigma: Residual SD of the market line on earlier seasons
            (evaluate.market_residual_sd), to turn a line into a probability.

    Returns:
        New DataFrame: the predictions plus home_points, away_points, margin, home_win,
        closing_spread, market_prob_home (from the closing line), market_saved_prob_home
        (from the line saved with the prediction), errors as margin minus spread for each
        model and the closing line, picked_winner (the ridge favorite won), and ats_result.
    """
    finals = results[["game_id", "home_points", "away_points", "margin", "home_win"]]
    lines = closing[["game_id", "market_spread"]].rename(
        columns={"market_spread": "closing_spread"}
    )
    scored = predictions.merge(finals, on="game_id", how="inner", validate="one_to_one")
    scored = scored.merge(lines, on="game_id", how="left", validate="one_to_one")

    scored["market_prob_home"] = evaluate.market_win_prob(scored["closing_spread"], market_sigma)
    scored["market_saved_prob_home"] = evaluate.market_win_prob(
        scored["market_spread"], market_sigma
    )
    for name in ["ridge", "huber"]:
        scored[f"{name}_error"] = scored["margin"] - scored[f"{name}_spread"]
    scored["closing_error"] = scored["margin"] - scored["closing_spread"]
    scored["picked_winner"] = (scored["ridge_home_win_prob"] > 0.5) == (scored["home_win"] == 1)
    scored["ats_result"] = [
        ats_result(m, c, y) if pd.notna(c) else "no line"
        for m, c, y in zip(scored["ridge_spread"], scored["closing_spread"], scored["margin"])
    ]
    return scored.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def summarize_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Model vs. market on the same games: MAE, log loss, Brier, and winners picked.

    Only games with both a closing line and a saved line are scored, so every row covers
    the same games.

    Args:
        scored: score_predictions output (one week, or several stacked).

    Returns:
        One row per model: ridge, huber, market (closing), market (when saved).
    """
    games = scored[scored["closing_spread"].notna() & scored["market_spread"].notna()]
    models = {
        "ridge": ("ridge_spread", "ridge_home_win_prob"),
        "huber": ("huber_spread", "huber_home_win_prob"),
        "market (closing)": ("closing_spread", "market_prob_home"),
        "market (when saved)": ("market_spread", "market_saved_prob_home"),
    }
    rows = {}
    for name, (spread_col, prob_col) in models.items():
        picked = (games[prob_col] > 0.5) == (games["home_win"] == 1)
        rows[name] = {
            "games": len(games),
            "mae": evaluate.mae(games["margin"], games[spread_col]),
            "log_loss": evaluate.log_loss(games["home_win"], games[prob_col]),
            "brier": evaluate.brier_score(games["home_win"], games[prob_col]),
            "winners_pct": float(picked.mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def biggest_misses(scored: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n games the ridge model missed by the most points, in readable form."""
    order = scored["ridge_error"].abs().sort_values(ascending=False).index[:n]
    rows = []
    for game in scored.loc[order].itertuples():
        rows.append(
            {
                "Matchup": f"{game.away_team} @ {game.home_team}",
                "Predicted line": format_line(game.ridge_spread, game.home_team, game.away_team),
                "Closing line": format_line(
                    game.closing_spread, game.home_team, game.away_team, exact=True
                ),
                "Final": f"{game.away_team} {game.away_points}, "
                f"{game.home_team} {game.home_points}",
                "Missed by": round(abs(game.ridge_error), 1),
            }
        )
    return pd.DataFrame(rows)
