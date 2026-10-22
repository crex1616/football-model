import pandas as pd

from football_model import clean


def make_raw_games() -> pd.DataFrame:
    """Three raw CFBD-style games: an FCS opponent, a neutral site, and one cancelled."""
    base = {column: None for column in clean.GAME_COLUMNS}
    rows = [
        {
            **base,
            "id": 2,
            "season": 2024,
            "week": 2,
            "seasonType": "regular",
            "startDate": "2024-09-07T19:30:00.000Z",
            "completed": True,
            "neutralSite": False,
            "homeTeam": "Ohio State",
            "homeClassification": "fbs",
            "homePoints": 56.0,
            "awayTeam": "Western Michigan",
            "awayClassification": "fbs",
            "awayPoints": 0.0,
            "homeLineScores": [21, 14, 14, 7],
            "excitementIndex": 1.2,
        },
        {
            **base,
            "id": 1,
            "season": 2024,
            "week": 1,
            "seasonType": "regular",
            "startDate": "2024-08-31T16:00:00.000Z",
            "completed": True,
            "neutralSite": True,
            "homeTeam": "Georgia",
            "homeClassification": "fbs",
            "homePoints": 34.0,
            "awayTeam": "Clemson",
            "awayClassification": "fbs",
            "awayPoints": 3.0,
            "homeLineScores": [7, 10, 7, 10],
            "excitementIndex": 2.5,
        },
        {
            **base,
            "id": 3,
            "season": 2024,
            "week": 1,
            "seasonType": "regular",
            "startDate": "2024-08-31T23:00:00.000Z",
            "completed": True,
            "neutralSite": False,
            "venueId": 3001,
            "homeId": 2006,
            "awayId": 2142,
            "homeTeam": "Akron",
            "homeClassification": "fbs",
            "homePoints": 10.0,
            "awayTeam": "Colgate",
            "awayClassification": "fcs",
            "awayPoints": 17.0,
            "homeLineScores": [0, 3, 7, 0],
            "excitementIndex": 4.0,
        },
        {
            **base,
            "id": 4,
            "season": 2024,
            "week": 5,
            "seasonType": "regular",
            "startDate": "2024-09-28T16:00:00.000Z",
            "completed": False,
            "neutralSite": False,
            "homeTeam": "App State",
            "homeClassification": "fbs",
            "homePoints": None,
            "awayTeam": "Liberty",
            "awayClassification": "fbs",
            "awayPoints": None,
            "homeLineScores": None,
            "excitementIndex": None,
        },
    ]
    return pd.DataFrame(rows)


def test_clean_games_drops_unfinished_and_leaky_columns():
    games = clean.clean_games(make_raw_games())

    assert 4 not in games["game_id"].values
    for leaky in ["homeLineScores", "home_line_scores", "excitementIndex", "excitement_index"]:
        assert leaky not in games.columns


def test_clean_games_margin_and_home_win():
    games = clean.clean_games(make_raw_games()).set_index("game_id")

    assert games.loc[2, "margin"] == 56
    assert games.loc[2, "home_win"] == 1
    assert games.loc[3, "margin"] == -7
    assert games.loc[3, "home_win"] == 0


def test_clean_games_sorted_by_kickoff_in_utc():
    games = clean.clean_games(make_raw_games())

    assert list(games["game_id"]) == [1, 3, 2]
    assert str(games["start_date"].dt.tz) == "UTC"


def test_clean_games_maps_fcs_and_keeps_neutral_flag():
    games = clean.clean_games(make_raw_games()).set_index("game_id")

    assert games.loc[3, "away_team"] == "FCS"
    assert games.loc[3, "home_team"] == "Akron"
    assert bool(games.loc[1, "neutral_site"]) is True
    assert bool(games.loc[2, "neutral_site"]) is False


def test_clean_games_keeps_ids_when_fcs_is_pooled():
    games = clean.clean_games(make_raw_games()).set_index("game_id")

    # Colgate becomes "FCS", but its CFBD id survives for location lookups.
    assert games.loc[3, "away_team"] == "FCS"
    assert games.loc[3, "away_id"] == 2142
    assert games.loc[3, "home_id"] == 2006
    assert games.loc[3, "venue_id"] == 3001


def test_clean_games_does_not_mutate_input():
    raw = make_raw_games()
    before = raw.copy()

    clean.clean_games(raw)

    pd.testing.assert_frame_equal(raw, before)


def test_consensus_lines_flips_sign_to_expected_home_margin():
    # Sportsbook style: home favored by 7 is printed as -7.
    raw_lines = pd.DataFrame({"id": [1], "provider": ["ESPN Bet"], "spread": [-7.0]})

    lines = clean.consensus_lines(raw_lines)

    assert lines.loc[0, "market_spread"] == 7.0


def test_consensus_lines_averages_providers_and_skips_missing():
    raw_lines = pd.DataFrame(
        {
            "id": [1, 1, 1, 2],
            "provider": ["ESPN Bet", "DraftKings", "Bovada", "ESPN Bet"],
            "spread": [-3.0, -4.0, None, 2.5],
        }
    )

    lines = clean.consensus_lines(raw_lines).set_index("game_id")

    assert lines.loc[1, "market_spread"] == 3.5
    assert lines.loc[1, "n_providers"] == 2
    assert lines.loc[2, "market_spread"] == -2.5


def test_add_market_lines_keeps_games_without_a_line():
    games = clean.clean_games(make_raw_games())
    lines = pd.DataFrame({"game_id": [1], "market_spread": [14.0], "n_providers": [3]})

    merged = clean.add_market_lines(games, lines).set_index("game_id")

    assert len(merged) == 3
    assert merged.loc[1, "market_spread"] == 14.0
    assert pd.isna(merged.loc[2, "market_spread"])
    assert merged.loc[2, "n_providers"] == 0
