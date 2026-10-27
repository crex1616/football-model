import pandas as pd
import pytest

from football_model import predict


def make_predictions(kickoffs: list[str]) -> pd.DataFrame:
    """Minimal prediction rows with the given UTC kickoffs."""
    return pd.DataFrame(
        {
            "game_id": range(1, len(kickoffs) + 1),
            "start_date": pd.to_datetime(kickoffs, utc=True),
            "ridge_spread": 3.0,
        }
    )


NOW = pd.Timestamp("2026-10-27 12:00", tz="UTC")


def test_save_refuses_to_overwrite(tmp_path):
    path = tmp_path / "week_09.csv"
    preds = make_predictions(["2026-10-31 16:00"])

    predict.save_predictions(preds, path, NOW)
    with pytest.raises(FileExistsError):
        predict.save_predictions(preds, path, NOW)


def test_save_refuses_games_already_started(tmp_path):
    preds = make_predictions(["2026-10-27 11:00", "2026-10-31 16:00"])

    with pytest.raises(ValueError):
        predict.save_predictions(preds, tmp_path / "week_09.csv", NOW)
    assert not (tmp_path / "week_09.csv").exists()


def test_drop_started_keeps_future_games_with_note(tmp_path):
    preds = make_predictions(["2026-10-27 11:00", "2026-10-31 16:00"])

    remaining = predict.drop_started(preds, NOW)
    predict.save_predictions(remaining, tmp_path / "week_09.csv", NOW)

    assert list(remaining["game_id"]) == [2]
    assert remaining["note"].str.contains("1 game").all()


def test_prediction_path():
    assert predict.prediction_path(2026, 9).name == "week_09.csv"
    assert predict.prediction_path(2026, 9).parent.name == "2026"


def test_format_line_names_the_favorite_with_a_minus():
    # Our sign: + = home favored. Sportsbook style names the favorite with a minus.
    assert predict.format_line(7.0, "Ohio State", "Michigan") == "Ohio State -7.0"
    assert predict.format_line(-6.5, "Cincinnati", "Utah") == "Utah -6.5"
    assert predict.format_line(0.0, "A", "B") == "Pick"
    assert predict.format_line(float("nan"), "A", "B") == "-"


def test_format_line_exact_keeps_quarter_points():
    assert predict.format_line(-5.75, "Fresno State", "Oregon State", exact=True) == (
        "Oregon State -5.75"
    )
    assert predict.format_line(7.0, "Duke", "Boston College", exact=True) == "Duke -7.0"
    assert predict.format_line(5.75, "A", "B") == "A -5.8"


def test_readable_table_kickoff_order_and_favorite_win_pct():
    preds = pd.DataFrame(
        {
            "game_id": [2, 1],
            "start_date": pd.to_datetime(["2026-10-31 23:30", "2026-10-31 16:00"], utc=True),
            "home_team": ["USC", "Duke"],
            "away_team": ["Ohio State", "Boston College"],
            "neutral_site": [False, True],
            "ridge_spread": [-12.9, 13.3],
            "ridge_home_win_prob": [0.223, 0.783],
            "market_spread": [-6.5, float("nan")],
        }
    )

    table = predict.readable_table(preds, timezone="America/Denver")

    assert list(table["Matchup"]) == ["Boston College vs Duke", "Ohio State @ USC"]
    assert list(table["Predicted line"]) == ["Duke -13.3", "Ohio State -12.9"]
    assert list(table["Market line"]) == ["-", "Ohio State -6.5"]
    assert list(table["Win %"]) == ["Duke 78%", "Ohio State 78%"]
    assert table.loc[0, "Kickoff"] == "Sat Oct 31 10:00 AM"
