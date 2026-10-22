import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from football_model import spread


def make_games(rows: list[tuple]) -> pd.DataFrame:
    """Build a tiny games table from (season, week, home, away, margin, neutral) tuples."""
    records = []
    for game_id, (season, week, home, away, margin, neutral) in enumerate(rows, start=1):
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
                "margin": margin,
                "neutral_site": neutral,
            }
        )
    return pd.DataFrame(records)


THREE_TEAMS = make_games(
    [
        (2024, 1, "A", "B", 10, False),
        (2024, 1, "C", "A", -3, True),
        (2024, 2, "B", "C", 7, False),
        (2024, 2, "A", "C", 14, False),
        (2024, 3, "C", "B", -4, False),
        (2024, 3, "B", "A", -6, True),
    ]
)


def zero_prior(teams: list[str]) -> pd.Series:
    return pd.Series(0.0, index=[spread.HOME_FIELD, *teams])


def test_design_matrix_entries():
    X, y = spread.design_matrix(THREE_TEAMS.iloc[:2], ["A", "B", "C"])

    # Game 1: A hosts B. Game 2: C "hosts" A at a neutral site.
    assert X.iloc[0].tolist() == [1.0, 1.0, -1.0, 0.0]
    assert X.iloc[1].tolist() == [0.0, -1.0, 0.0, 1.0]
    assert y.tolist() == [10.0, -3.0]


def test_near_zero_alpha_matches_least_squares():
    teams = ["A", "B", "C"]
    params = spread.SpreadParams(alpha=1e-8)
    fit = spread.fit_ratings(THREE_TEAMS, zero_prior(teams), params)

    X, y = spread.design_matrix(THREE_TEAMS, teams)
    # Team columns are collinear (each row sums to 0), so OLS has many solutions;
    # lstsq returns the minimum-norm one, which is the limit of ridge as alpha -> 0.
    ols, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
    assert fit.to_numpy() == pytest.approx(ols, abs=1e-5)


def test_zero_prior_matches_sklearn_ridge():
    teams = ["A", "B", "C"]
    params = spread.SpreadParams(alpha=5.0)
    fit = spread.fit_ratings(THREE_TEAMS, zero_prior(teams), params)

    X, y = spread.design_matrix(THREE_TEAMS, teams)
    sk = Ridge(alpha=5.0, fit_intercept=False).fit(X, y)
    assert fit.to_numpy() == pytest.approx(sk.coef_, abs=1e-8)


def test_large_alpha_returns_prior():
    prior = pd.Series({spread.HOME_FIELD: 2.5, "A": 7.0, "B": -3.0, "C": 0.0})
    fit = spread.fit_ratings(THREE_TEAMS, prior, spread.SpreadParams(alpha=1e9))
    assert fit.to_numpy() == pytest.approx(prior.to_numpy(), abs=1e-6)


def test_spread_sign_home_favorite_is_positive():
    # Season 1: A beats B by 7 every week at neutral sites. Season 2's week-1 game has
    # no current-season data, so it uses the prior: A rated above B, spread > 0.
    games = make_games(
        [(2023, w, "A", "B", 7, True) for w in range(1, 6)] + [(2024, 1, "A", "B", 0, True)]
    )
    params = spread.SpreadParams(alpha=1e-6, revert=0.0)
    pred = spread.run_spread(games, params).iloc[-1]

    assert pred["hfa_pre"] == 0.0
    assert pred["spread_pred"] == pytest.approx(7.0, abs=1e-3)
    assert pred["spread_pred"] == pred["home_rating_pre"] - pred["away_rating_pre"]


def test_first_week_uses_prior_only():
    pred = spread.run_spread(THREE_TEAMS, spread.SpreadParams())
    week1 = pred[pred["week"] == 1]
    assert (week1["games_in_fit"] == 0).all()
    assert (week1["spread_pred"] == 0.0).all()


def test_no_leakage_from_later_games():
    params = spread.SpreadParams(alpha=1.0)
    base = spread.run_spread(THREE_TEAMS, params)

    # Change a week-2 result. Predictions for weeks 1-2 must not move; week 3 should.
    changed = THREE_TEAMS.copy()
    changed.loc[changed["game_id"] == 3, "margin"] = -40
    after = spread.run_spread(changed, params)

    early = base["week"] <= 2
    assert after.loc[early, "spread_pred"].tolist() == base.loc[early, "spread_pred"].tolist()
    assert not np.allclose(after.loc[~early, "spread_pred"], base.loc[~early, "spread_pred"])


def test_margin_cap_changes_fit_not_stored_margin():
    games = make_games([(2024, 1, "A", "B", 60, False), (2024, 2, "A", "B", 0, False)])
    teams = ["A", "B"]
    uncapped = spread.fit_ratings(games.iloc[:1], zero_prior(teams), spread.SpreadParams(1.0))
    capped = spread.fit_ratings(
        games.iloc[:1], zero_prior(teams), spread.SpreadParams(1.0, margin_cap=21)
    )
    assert capped["A"] < uncapped["A"]

    pred = spread.run_spread(games, spread.SpreadParams(1.0, margin_cap=21))
    assert pred["margin"].tolist() == [60, 0]


def test_recency_weights_half_life():
    as_of = pd.Timestamp("2024-10-01", tz="UTC")
    dates = pd.Series([as_of, as_of - pd.Timedelta(weeks=2), as_of - pd.Timedelta(weeks=4)])
    w = spread.recency_weights(dates, as_of, half_life_weeks=2)
    assert w == pytest.approx([1.0, 0.5, 0.25])


def test_season_prior_reverts_and_handles_new_teams():
    last = pd.Series({spread.HOME_FIELD: 2.5, "A": 12.0, "B": 0.0, "FCS": -20.0})
    prior = spread.season_prior(last, ["A", "B", "FCS", "New"], revert=0.5)

    # FBS mean of A and B is 6, so centered A = 6, B = -6, FCS = -26; FBS halved.
    assert prior[spread.HOME_FIELD] == 2.5
    assert prior["A"] == pytest.approx(3.0)
    assert prior["B"] == pytest.approx(-3.0)
    assert prior["FCS"] == pytest.approx(-26.0)
    assert prior["New"] == pytest.approx(-26.0)


def test_default_params_are_plain_ridge():
    assert spread.SpreadParams().huber_c is None
    assert spread.HUBER_PARAMS.huber_c is not None


def test_huber_weights_hand_calculation():
    # c = 10: residuals inside c keep weight 1; residual 20 (= 2c) gets 1/2; -40 gets 1/4.
    w = spread.huber_weights(np.array([0.0, 5.0, -10.0, 20.0, -40.0]), c=10.0)
    assert w == pytest.approx([1.0, 1.0, 1.0, 0.5, 0.25])


def test_huber_with_huge_c_equals_ridge():
    teams = ["A", "B", "C"]
    ridge = spread.fit_ratings(THREE_TEAMS, zero_prior(teams), spread.SpreadParams(alpha=2.0))
    huber = spread.fit_ratings(
        THREE_TEAMS, zero_prior(teams), spread.SpreadParams(alpha=2.0, huber_c=1e6)
    )
    assert huber.to_numpy() == pytest.approx(ridge.to_numpy())


def test_huber_discounts_a_blowout():
    # Four ordinary results plus one 70-point blowout by A. Under squared loss the
    # blowout drags A's rating up a lot; under Huber it counts only as a c-point surprise.
    games = make_games(
        [
            (2024, 1, "A", "B", 7, True),
            (2024, 1, "C", "D", 3, True),
            (2024, 2, "B", "C", 4, True),
            (2024, 2, "D", "A", -6, True),
            (2024, 3, "A", "C", 70, True),
        ]
    )
    teams = ["A", "B", "C", "D"]
    ridge = spread.fit_ratings(games, zero_prior(teams), spread.SpreadParams(alpha=0.5))
    huber = spread.fit_ratings(games, zero_prior(teams), spread.SpreadParams(alpha=0.5, huber_c=7))
    assert huber["A"] - huber["C"] < ridge["A"] - ridge["C"]


def test_huber_does_not_cap_predictions():
    # A beats B by exactly 40 every week at neutral sites. Every residual is near 0, so
    # Huber matches ridge and the next season's prior spread is still ~40 (a margin cap
    # of 14 would cap it at 14).
    games = make_games(
        [(2023, w, "A", "B", 40, True) for w in range(1, 6)] + [(2024, 1, "A", "B", 0, True)]
    )
    params = spread.SpreadParams(alpha=1e-6, revert=0.0, huber_c=14.0)
    pred = spread.run_spread(games, params).iloc[-1]
    assert pred["spread_pred"] == pytest.approx(40.0, abs=1e-2)


def test_huber_no_leakage_from_later_games():
    params = spread.SpreadParams(alpha=1.0, huber_c=5.0)
    base = spread.run_spread(THREE_TEAMS, params)
    changed = THREE_TEAMS.copy()
    changed.loc[changed["game_id"] == 3, "margin"] = -40
    after = spread.run_spread(changed, params)

    early = base["week"] <= 2
    assert after.loc[early, "spread_pred"].tolist() == base.loc[early, "spread_pred"].tolist()
