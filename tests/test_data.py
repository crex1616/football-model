import cfbd
import pandas as pd

from football_model import data


def test_cached_fetches_once_then_reads_file(tmp_path):
    path = tmp_path / "games_2024.parquet"
    calls = []

    def fake_fetch():
        calls.append(1)
        return pd.DataFrame({"id": [1, 2], "homePoints": [21, 14]})

    first = data._cached(path, fake_fetch)
    second = data._cached(path, fake_fetch)

    assert len(calls) == 1
    assert path.exists()
    pd.testing.assert_frame_equal(first, second)


def test_to_records_converts_dates_and_enums_to_plain_values():
    game = cfbd.Game.from_dict(
        {
            "id": 1,
            "season": 2024,
            "week": 1,
            "seasonType": "regular",
            "startDate": "2024-08-24T16:00:00.000Z",
            "startTimeTBD": False,
            "completed": True,
            "neutralSite": True,
            "conferenceGame": False,
            "homeId": 1,
            "homeTeam": "Georgia",
            "awayId": 2,
            "awayTeam": "Clemson",
        }
    )

    record = data._to_records([game])[0]

    assert record["seasonType"] == "regular"
    assert record["startDate"] == "2024-08-24T16:00:00+00:00"


def test_flatten_lines_one_row_per_provider():
    records = [
        {
            "id": 10,
            "homeTeam": "Georgia",
            "awayTeam": "Clemson",
            "lines": [
                {"provider": "Bovada", "spread": -13.5},
                {"provider": "DraftKings", "spread": -13.0},
            ],
        },
        {"id": 11, "homeTeam": "Ohio State", "awayTeam": "Akron", "lines": []},
    ]

    df = data.flatten_lines(records)

    assert len(df) == 2
    assert set(df["provider"]) == {"Bovada", "DraftKings"}
    assert (df["id"] == 10).all()
    assert "lines" not in df.columns
