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


def test_flatten_records_joins_nested_keys():
    records = [
        {"gameId": 1, "team": "A", "offense": {"ppa": 0.2, "successRate": 0.45}},
        {"gameId": 1, "team": "B", "offense": {"ppa": -0.1, "successRate": 0.38}},
    ]
    df = data.flatten_records(records)

    assert list(df.columns) == ["gameId", "team", "offense.ppa", "offense.successRate"]
    assert df["offense.ppa"].tolist() == [0.2, -0.1]


def test_flatten_records_empty_response():
    assert data.flatten_records([]).empty


def test_phase5_loaders_use_their_cache_files(monkeypatch):
    seen = []

    def fake_cached(path, fetch_fn):
        seen.append(path.name)
        return pd.DataFrame()

    monkeypatch.setattr(data, "_cached", fake_cached)
    data.load_advanced_game_stats(2024)
    data.load_recruiting(2024)
    data.load_talent(2024)
    data.load_returning_production(2024)
    data.load_venues()
    data.load_teams()

    assert seen == [
        "advanced_game_stats_2024.parquet",
        "recruiting_2024.parquet",
        "talent_2024.parquet",
        "returning_production_2024.parquet",
        "venues.parquet",
        "teams.parquet",
    ]


def test_cached_refetches_stale_week_until_final(tmp_path):
    path = tmp_path / "games_2026_regular_week_09.parquet"
    responses = [
        pd.DataFrame({"id": [1], "completed": [False]}),
        pd.DataFrame({"id": [1], "completed": [True]}),
    ]
    calls = []

    def fake_fetch():
        calls.append(1)
        return responses[len(calls) - 1]

    data._cached(path, fake_fetch, is_stale=data.week_has_unfinished_games)
    data._cached(path, fake_fetch, is_stale=data.week_has_unfinished_games)
    final = data._cached(path, fake_fetch, is_stale=data.week_has_unfinished_games)

    assert len(calls) == 2
    assert bool(final["completed"].all())


def test_week_lines_not_final():
    assert data.week_lines_not_final(pd.DataFrame())
    assert data.week_lines_not_final(pd.DataFrame({"homeScore": [21.0, None]}))
    assert not data.week_lines_not_final(pd.DataFrame({"homeScore": [21.0, 14.0]}))
