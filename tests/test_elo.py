import math

import pandas as pd
import pytest

from football_model import elo


def make_games(rows: list[tuple]) -> pd.DataFrame:
    """Build a tiny games table from (season, day, home, away, margin, neutral) tuples."""
    records = []
    for game_id, (season, day, home, away, margin, neutral) in enumerate(rows, start=1):
        records.append(
            {
                "game_id": game_id,
                "season": season,
                "start_date": pd.Timestamp(f"{season}-09-01", tz="UTC") + pd.Timedelta(days=day),
                "home_team": home,
                "away_team": away,
                "margin": margin,
                "neutral_site": neutral,
            }
        )
    return pd.DataFrame(records)


def test_expected_home_win_hand_calculation():
    # 100-point edge, no home field: 1 / (1 + 10^(-100/400)) = 0.6401
    assert elo.expected_home_win(1600, 1500, home_field=0) == pytest.approx(0.6401, abs=1e-4)


def test_neutral_site_ignores_home_field():
    neutral = elo.expected_home_win(1500, 1500, home_field=70, neutral=True)
    home = elo.expected_home_win(1500, 1500, home_field=70, neutral=False)
    assert neutral == pytest.approx(0.5)
    assert home > 0.5


def test_single_update_hand_calculation():
    # Both teams start at 1500, so E = 0.5. Home wins: shift = 20 * (1 - 0.5) = 10.
    params = elo.EloParams(k=20, home_field=0, use_mov=False)
    rated = elo.run_elo(make_games([(2024, 0, "A", "B", 7, False)]), params)

    assert rated.loc[0, "elo_prob_home"] == pytest.approx(0.5)
    assert rated.loc[0, "home_elo_post"] == pytest.approx(1510)
    assert rated.loc[0, "away_elo_post"] == pytest.approx(1490)


def test_update_with_rating_gap_hand_calculation():
    # After game 1: A = 1510, B = 1490. Game 2: A hosts B again, H = 0, A wins.
    # E = 1 / (1 + 10^(-20/400)) = 0.52876; shift = 20 * (1 - 0.52876) = 9.4248.
    params = elo.EloParams(k=20, home_field=0, use_mov=False)
    games = make_games([(2024, 0, "A", "B", 7, False), (2024, 7, "A", "B", 3, False)])
    second = elo.run_elo(games, params).iloc[1]

    expected = 1 / (1 + 10 ** (-20 / 400))
    assert second["elo_prob_home"] == pytest.approx(expected)
    assert second["home_elo_post"] == pytest.approx(1510 + 20 * (1 - expected))
    assert second["away_elo_post"] == pytest.approx(1490 - 20 * (1 - expected))


def test_mov_multiplier_hand_calculation():
    assert elo.mov_multiplier(14, 100) == pytest.approx(math.log(15) * 2.2 / 2.3)
    # Upsets (negative winner diff) get a bigger multiplier than expected wins.
    assert elo.mov_multiplier(14, -100) > elo.mov_multiplier(14, 100)


def test_ratings_are_zero_sum():
    params = elo.EloParams(k=25, home_field=60, use_mov=True)
    games = make_games(
        [
            (2024, 0, "A", "B", 21, False),
            (2024, 1, "C", "D", -3, True),
            (2024, 7, "A", "C", 10, False),
            (2024, 8, "B", "D", -14, False),
        ]
    )
    rated = elo.run_elo(games, params)
    ratings = elo.final_ratings(rated)

    assert ratings["rating"].sum() == pytest.approx(4 * 1500)


def test_pre_game_ratings_ignore_later_games():
    """Leakage check: changing a later result must not change earlier pre-game ratings."""
    params = elo.EloParams()
    rows = [
        (2024, 0, "A", "B", 21, False),
        (2024, 7, "B", "C", 3, False),
        (2024, 14, "C", "A", -10, False),
    ]
    changed = rows[:2] + [(2024, 14, "C", "A", 35, False)]

    original = elo.run_elo(make_games(rows), params)
    altered = elo.run_elo(make_games(changed), params)

    pre_cols = ["home_elo_pre", "away_elo_pre", "elo_prob_home"]
    pd.testing.assert_frame_equal(original[pre_cols], altered[pre_cols])
    assert original.loc[2, "home_elo_post"] != altered.loc[2, "home_elo_post"]


def test_season_reversion_toward_fbs_mean():
    # Force A = 1700, B = 1300 by setting K so one game moves 200: E = 0.5, shift = K/2.
    params = elo.EloParams(k=400, home_field=0, revert=0.25, use_mov=False)
    games = make_games([(2023, 0, "A", "B", 7, False), (2024, 0, "A", "B", 7, True)])
    second = elo.run_elo(games, params).iloc[1]

    # FBS mean is 1500, so A reverts to 1500 + 0.75 * 200 = 1650 and B to 1350.
    assert second["home_elo_pre"] == pytest.approx(1650)
    assert second["away_elo_pre"] == pytest.approx(1350)


def test_new_program_starts_at_fcs_rating():
    params = elo.EloParams(k=40, home_field=0, revert=0.0, use_mov=False)
    games = make_games(
        [
            (2023, 0, "A", "FCS", 30, False),
            (2024, 0, "A", "NewU", 10, False),
        ]
    )
    rated = elo.run_elo(games, params)

    fcs_rating = rated.loc[0, "away_elo_post"]
    assert fcs_rating < 1500
    assert rated.loc[1, "away_elo_pre"] == pytest.approx(fcs_rating)


def test_run_elo_does_not_mutate_input():
    games = make_games([(2024, 7, "A", "B", 7, False), (2024, 0, "C", "D", 3, False)])
    before = games.copy()

    elo.run_elo(games, elo.EloParams())

    pd.testing.assert_frame_equal(games, before)


def test_upcoming_elo_diff_uses_latest_ratings():
    played = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2024],
            "start_date": [pd.Timestamp("2024-09-01", tz="UTC")],
            "home_team": ["A"],
            "away_team": ["B"],
            "neutral_site": [False],
            "margin": [10],
        }
    )
    upcoming = pd.DataFrame(
        {
            "season": [2024, 2024],
            "start_date": [pd.Timestamp("2024-09-08", tz="UTC")] * 2,
            "home_team": ["B", "A"],
            "away_team": ["A", "Newcomer"],
            "neutral_site": [True, False],
        }
    )
    params = elo.EloParams()

    diff = elo.upcoming_elo_diff(played, upcoming, params)

    rated = elo.run_elo(played, params)
    a_post, b_post = rated.loc[0, "home_elo_post"], rated.loc[0, "away_elo_post"]
    assert diff.iloc[0] == pytest.approx(b_post - a_post)  # neutral: no home field
    # No FCS rating exists here, so a never-rated team starts at the initial rating.
    assert diff.iloc[1] == pytest.approx(a_post - params.initial + params.home_field)
