import numpy as np
import pandas as pd
import pytest

from football_model import features


def make_games(rows: list[tuple]) -> pd.DataFrame:
    """Games from (season, week, home, away, home_epa, away_epa, neutral) tuples.

    Success rate is set equal to EPA so both metrics can run; ids are fixed per team.
    """
    ids = {"A": 1, "B": 2, "C": 3, "FCS": 99}
    records = []
    for game_id, (season, week, home, away, h_epa, a_epa, neutral) in enumerate(rows, start=1):
        records.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "season_type": "regular",
                "start_date": pd.Timestamp(f"{season}-09-01", tz="UTC")
                + pd.Timedelta(weeks=week - 1, hours=game_id),
                "home_team": home,
                "away_team": away,
                "home_id": ids[home],
                "away_id": ids[away],
                "neutral_site": neutral,
                "home_epa": h_epa,
                "away_epa": a_epa,
                "home_sr": h_epa,
                "away_sr": a_epa,
            }
        )
    return pd.DataFrame(records)


THREE_TEAMS = make_games(
    [
        (2024, 1, "A", "B", 0.40, 0.00, False),
        (2024, 1, "C", "A", 0.05, 0.35, True),
        (2024, 2, "B", "C", 0.10, 0.10, False),
        (2024, 2, "A", "C", 0.45, -0.05, False),
        (2024, 3, "C", "B", 0.00, 0.15, False),
        (2024, 3, "B", "A", 0.05, 0.30, True),
    ]
)


# --- Efficiency ratings -------------------------------------------------------------


def test_design_matrix_offense_plus_defense_minus():
    rows = features.offense_rows(THREE_TEAMS.iloc[[0]], "epa")
    X, y = features.efficiency_design_matrix(rows, ["A", "B", "C"])

    # Row 0: A's offense at home vs. B's defense.
    assert X.loc[0, "const"] == 1.0
    assert X.loc[0, "home_field"] == 1.0
    assert X.loc[0, "O:A"] == 1.0
    assert X.loc[0, "D:B"] == -1.0
    # Row 1: B's offense on the road vs. A's defense.
    assert X.loc[1, "home_field"] == -1.0
    assert X.loc[1, "O:B"] == 1.0
    assert X.loc[1, "D:A"] == -1.0
    assert list(y) == [0.40, 0.00]


def test_neutral_site_has_no_home_term():
    rows = features.offense_rows(THREE_TEAMS.iloc[[1]], "epa")
    assert list(rows["home"]) == [0.0, 0.0]


def test_near_zero_alpha_matches_least_squares():
    rows = features.offense_rows(THREE_TEAMS, "epa")
    teams = ["A", "B", "C"]
    prior = features.efficiency_prior(None, teams, 0.3)

    fit = features.fit_efficiency(rows, teams, prior, alpha=1e-8)

    X, y = features.efficiency_design_matrix(rows, teams)
    # The model is not identified (shifting every O and D together changes nothing), so
    # compare fitted values, which are unique, rather than coefficients.
    ols_fitted = X.to_numpy() @ np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(X.to_numpy() @ fit.to_numpy(), ols_fitted, atol=1e-5)


def test_strongest_offense_gets_largest_rating():
    rows = features.offense_rows(THREE_TEAMS, "epa")
    teams = ["A", "B", "C"]
    fit = features.fit_efficiency(rows, teams, features.efficiency_prior(None, teams, 0.3), 1.0)

    assert fit["O:A"] > fit["O:B"]
    assert fit["O:A"] > fit["O:C"]


def test_efficiency_ratings_ignore_later_games():
    params = features.EfficiencyParams(alpha=1.0, revert=0.3)
    before = features.run_efficiency(THREE_TEAMS, "epa", params)

    changed = THREE_TEAMS.copy()
    changed.loc[changed["week"] == 3, ["home_epa", "away_epa"]] = [2.0, -2.0]
    after = features.run_efficiency(changed, "epa", params)

    early = THREE_TEAMS["week"] <= 3  # week 3's ratings come from weeks 1-2 only
    pd.testing.assert_frame_equal(before[early], after[early])


def test_first_week_uses_prior_only():
    ratings = features.run_efficiency(THREE_TEAMS, "epa", features.EfficiencyParams())
    week1 = THREE_TEAMS["week"] == 1
    assert (ratings.loc[week1].drop(columns=["epa_const", "epa_hfa"]) == 0.0).all().all()


def test_efficiency_edge_is_home_minus_away():
    ratings = pd.DataFrame(
        {
            "epa_home_off": [0.2],
            "epa_home_def": [0.1],
            "epa_away_off": [0.05],
            "epa_away_def": [0.0],
        }
    )
    assert features.efficiency_edge(ratings, "epa").iloc[0] == pytest.approx(0.25)


def test_game_efficiency_matches_pooled_fcs_by_elimination():
    games = pd.DataFrame({"game_id": [1], "home_team": ["A"], "away_team": ["FCS"]})
    stats = pd.DataFrame(
        {
            "gameId": [1, 1],
            "team": ["A", "Colgate"],
            "offense.ppa": [0.5, -0.2],
            "offense.successRate": [0.55, 0.30],
        }
    )

    out = features.game_efficiency(stats, games)

    assert out.loc[0, "home_epa"] == 0.5
    assert out.loc[0, "away_epa"] == -0.2
    assert out.loc[0, "away_sr"] == 0.30


# --- Counts, rest, travel -----------------------------------------------------------


def test_games_played_counts_only_earlier_games():
    played = features.games_played_before(THREE_TEAMS)

    # Game 5: C already played games 2, 3, 4. Game 6: B already played games 1, 3, 5.
    assert list(played["home_played"]) == [0, 0, 1, 2, 3, 3]
    assert list(played["away_played"]) == [0, 1, 1, 2, 2, 3]


def test_games_played_ignores_later_games():
    before = features.games_played_before(THREE_TEAMS.iloc[:4])
    after = features.games_played_before(THREE_TEAMS).iloc[:4]
    pd.testing.assert_frame_equal(before, after)


def test_rest_days_counted_and_capped():
    rest = features.rest_days(THREE_TEAMS)

    # Game 1 is each team's first game: the cap.
    assert rest.loc[0, "home_rest"] == 14
    # Game 3 (week 2): B last played in game 1, 7 days and 2 hours earlier.
    assert rest.loc[2, "home_rest"] == pytest.approx(7 + 2 / 24)
    # A three-week gap is capped at 14.
    gap = make_games([(2024, 1, "A", "B", 0, 0, False), (2024, 4, "A", "C", 0, 0, False)])
    assert features.rest_days(gap).loc[1, "home_rest"] == 14


def test_rest_days_ignore_later_games():
    before = features.rest_days(THREE_TEAMS.iloc[:4])
    after = features.rest_days(THREE_TEAMS).iloc[:4]
    pd.testing.assert_frame_equal(before, after)


def test_haversine_known_distance():
    # Los Angeles (34.05, -118.24) to New York (40.71, -74.01): about 2,445 miles.
    miles = features.haversine_miles(34.05, -118.24, 40.71, -74.01)
    assert miles == pytest.approx(2445, abs=10)
    assert features.haversine_miles(30.0, -90.0, 30.0, -90.0) == 0.0


def test_travel_zero_at_own_stadium():
    games = pd.DataFrame({"home_id": [1], "away_id": [2], "venue_id": [10]})
    teams = pd.DataFrame(
        {"id": [1, 2], "location.latitude": [40.0, 34.0], "location.longitude": [-83.0, -118.0]}
    )
    venues = pd.DataFrame({"id": [10], "latitude": [40.0], "longitude": [-83.0]})

    travel = features.travel_miles(games, teams, venues)

    assert travel.loc[0, "home_travel"] == 0.0
    assert travel.loc[0, "away_travel"] > 1500


# --- Preseason ----------------------------------------------------------------------


def test_returning_ppa_clips_and_fills_missing_with_season_mean():
    games = pd.DataFrame(
        {"season": [2024, 2024], "home_team": ["A", "C"], "away_team": ["B", "FCS"]}
    )
    returning = pd.DataFrame(
        {"season": [2024, 2024, 2024], "team": ["A", "B", "C"], "percentPPA": [0.8, -3.0, 0.5]}
    )

    out = features.returning_ppa(games, returning)

    assert out.loc[0, "home_ret_ppa"] == 0.8
    assert out.loc[0, "away_ret_ppa"] == 0.0  # -3.0 clipped to 0
    assert out.loc[1, "away_ret_ppa"] == pytest.approx((0.8 + 0.0 + 0.5) / 3)  # FCS: mean


# --- Features for games not yet played ----------------------------------------------


def upcoming_inputs(games: pd.DataFrame) -> dict:
    """Hand-built stats, returning production, team and venue locations for THREE_TEAMS."""
    stats = pd.concat(
        [
            games[["game_id", "home_team", "home_epa"]].set_axis(["gameId", "team", "v"], axis=1),
            games[["game_id", "away_team", "away_epa"]].set_axis(["gameId", "team", "v"], axis=1),
        ]
    ).dropna()
    stats = stats.assign(**{"offense.ppa": stats["v"], "offense.successRate": stats["v"]})
    return {
        "stats": stats.drop(columns="v"),
        "returning": pd.DataFrame(
            {"season": 2024, "team": ["A", "B", "C"], "percentPPA": [0.8, 0.5, 0.3]}
        ),
        "teams": pd.DataFrame(
            {
                "id": [1, 2, 3],
                "location.latitude": [40.0, 35.0, 30.0],
                "location.longitude": [-83.0, -90.0, -97.0],
            }
        ),
        "venues": pd.DataFrame(
            {"id": [1, 2, 3], "latitude": [40.0, 35.0, 30.0], "longitude": [-83.0, -90.0, -97.0]}
        ),
    }


def with_venues(games: pd.DataFrame) -> pd.DataFrame:
    """Each game is at the home team's stadium (venue id = team id)."""
    return games.assign(venue_id=games["home_id"])


def test_upcoming_features_match_backtest_features():
    games = with_venues(THREE_TEAMS)
    inputs = upcoming_inputs(games)
    week3 = games["week"] == 3
    upcoming = games[week3].drop(columns=["home_epa", "away_epa", "home_sr", "away_sr"])

    live = features.upcoming_features(
        games[~week3],
        upcoming,
        inputs["stats"],
        inputs["returning"],
        inputs["teams"],
        inputs["venues"],
        {m: features.EfficiencyParams(1.0, 0.3) for m in features.METRICS},
    )

    # Backtest path: the same functions run on the full table.
    params = features.EfficiencyParams(1.0, 0.3)
    edge = features.efficiency_edge(features.run_efficiency(games, "epa", params), "epa")
    full = features.add_preseason_and_situational(
        games, inputs["returning"], inputs["teams"], inputs["venues"]
    )
    np.testing.assert_allclose(live["epa_edge"], edge[week3])
    for col in ["ret_ppa_diff", "ret_ppa_diff_faded", "rest_diff", "travel_diff"]:
        np.testing.assert_allclose(live[col], full.loc[week3.to_numpy(), col])


def test_upcoming_features_ignore_games_after_kickoff():
    games = with_venues(THREE_TEAMS)
    inputs = upcoming_inputs(games)
    week2 = games["week"] == 2
    upcoming = games[week2].drop(columns=["home_epa", "away_epa", "home_sr", "away_sr"])
    params = {m: features.EfficiencyParams(1.0, 0.3) for m in features.METRICS}
    args = (inputs["stats"], inputs["returning"], inputs["teams"], inputs["venues"], params)

    only_before = features.upcoming_features(games[games["week"] == 1], upcoming, *args)
    with_later = features.upcoming_features(games[~week2], upcoming, *args)  # includes week 3

    pd.testing.assert_frame_equal(only_before, with_later)
