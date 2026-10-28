import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from football_model import data, rankings, simulate


def game(home, away, margin=None, neutral=False):
    row = {"home_team": home, "away_team": away, "neutral_site": neutral}
    if margin is not None:
        row["margin"] = margin
    return row


def ratings(values: dict[str, float], home_field: float = 0.0) -> pd.Series:
    return pd.Series({rankings.HOME_FIELD: home_field, **values})


def test_win_distribution_by_hand():
    np.testing.assert_allclose(rankings.win_distribution(np.array([0.5, 0.5])), [0.25, 0.5, 0.25])
    # p = (1, 0, 0.3): one sure win, one sure loss, one 30% game.
    np.testing.assert_allclose(
        rankings.win_distribution(np.array([1.0, 0.0, 0.3])), [0, 0.7, 0.3, 0]
    )
    dist = rankings.win_distribution(np.array([0.9, 0.2, 0.55, 0.7, 0.1]))
    assert dist.sum() == pytest.approx(1.0)
    assert (dist @ np.arange(6)) == pytest.approx(0.9 + 0.2 + 0.55 + 0.7 + 0.1)  # mean = sum of p


def test_harder_schedule_ranks_higher_for_same_record():
    # Hard and Easy both go 3-0, Hard against strong teams.
    mean = ratings(
        {"Hard": 10, "Easy": 10, "S1": 15, "S2": 15, "S3": 15, "W1": -15, "W2": -15, "W3": -15}
    )
    games = pd.DataFrame(
        [game("Hard", t, 3, neutral=True) for t in ["S1", "S2", "S3"]]
        + [game("Easy", t, 3, neutral=True) for t in ["W1", "W2", "W3"]]
    )
    conferences = {t: "X" for t in mean.index.drop(rankings.HOME_FIELD)}

    table = rankings.resume_table(games, mean, 14.0, conferences, top_n=2).set_index("team")

    assert table.loc["Hard", "sor"] < table.loc["Easy", "sor"]
    assert table.loc["Hard", "resume_rank"] < table.loc["Easy", "resume_rank"]
    assert table.loc["Hard", "schedule_strength"] < table.loc["Easy", "schedule_strength"]
    assert table.loc["Hard", "wins"] == 3 and table.loc["Hard", "losses"] == 0


def test_sor_matches_hand_calculation():
    # Benchmark = mean of top 2 = (10 + 6) / 2 = 8. A plays B at home (hfa 2), then C away.
    mean = ratings({"A": 10, "B": 6, "C": 0}, home_field=2.0)
    games = pd.DataFrame([game("A", "B", 7), game("C", "A", 3)])  # A wins, A loses
    conferences = {"A": "X", "B": "X", "C": "X"}

    table = rankings.resume_table(games, mean, 10.0, conferences, top_n=2).set_index("team")

    p1 = norm.cdf((8 - 6 + 2) / 10)  # vs. B at home
    p2 = norm.cdf((8 - 0 - 2) / 10)  # at C
    expected = 1 - (1 - p1) * (1 - p2)  # P(at least 1 win)
    assert table.loc["A", "sor"] == pytest.approx(expected)
    assert table.loc["A", "benchmark_wins"] == pytest.approx(p1 + p2)


def test_resume_uses_only_the_games_it_is_given():
    """Future games (no margin, not passed in) can't change a résumé."""
    mean = ratings({"A": 5, "B": 0, "C": -5})
    conferences = {"A": "X", "B": "X", "C": "X"}
    played = pd.DataFrame([game("A", "B", 7), game("B", "C", 3)])

    before = rankings.resume_table(played, mean, 14.0, conferences, top_n=2)
    # Same games, after a later week's results are known: SOR for these games is the same.
    later = pd.concat([played, pd.DataFrame([game("C", "A", 20)])], ignore_index=True)
    after_week = rankings.resume_table(later.iloc[:2], mean, 14.0, conferences, top_n=2)

    pd.testing.assert_frame_equal(before, after_week)


def test_benchmark_is_mean_of_top_n():
    mean = ratings({"A": 9, "B": 6, "C": 3, "D": 0, "FCS": -30})

    assert rankings.benchmark_rating(mean, ["A", "B", "C", "D"], top_n=2) == pytest.approx(7.5)


def test_power_table_order_and_intervals():
    mean = ratings({"A": 1.0, "B": 5.0, "C": 3.0})
    ucov = np.diag([0.1, 0.01, 0.04, 0.09])

    table = rankings.power_table(mean, ucov, 10.0, {"A": "X", "B": "X", "C": "X"})

    assert list(table["team"]) == ["B", "C", "A"]
    assert list(table["power_rank"]) == [1, 2, 3]
    np.testing.assert_allclose(table["rating_se"], 10.0 * np.sqrt([0.04, 0.09, 0.01]))
    width = table["rating_high"] - table["rating_low"]
    np.testing.assert_allclose(width, 2 * rankings.INTERVAL_Z * table["rating_se"])


def test_schedule_tails_match_win_distribution():
    mean = ratings({"A": 5, "B": 0, "C": -5}, home_field=2.0)
    schedule = pd.DataFrame([game("A", "B"), game("C", "A")])  # unplayed: no margin

    tails = rankings.schedule_tails(schedule, ["A"], mean, 10.0, benchmark=5.0, max_wins=3)

    p = norm.cdf(np.array([(5 - 0 + 2) / 10, (5 + 5 - 2) / 10]))
    dist = rankings.win_distribution(p)
    np.testing.assert_allclose(tails[0], [1.0, dist[1:].sum(), dist[2], 0.0])


def test_compare_to_poll():
    ranks = pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})
    poll = pd.Series({"A": 2.0, "B": 1.0, "C": 3.0, "E": 4.0})

    result = rankings.compare_to_poll(ranks, poll)

    assert result["top4"] == 0.75  # poll top 4 = A, B, C, E; ours = A, B, C, D
    assert result["mean_abs_rank_diff"] == pytest.approx((1 + 1 + 0 + 1) / 4)
    assert 0 < result["spearman"] < 1


def test_flatten_polls_and_poll_ranks():
    records = [
        {
            "season": 2024,
            "seasonType": "regular",
            "week": 16,
            "polls": [
                {
                    "poll": rankings.COMMITTEE_POLL,
                    "isFinal": True,
                    "ranks": [
                        {"rank": 1, "school": "Oregon", "teamId": 1},
                        {"rank": 2, "school": "Georgia", "teamId": 2},
                    ],
                }
            ],
        }
    ]

    polls = data.flatten_polls(records)

    assert len(polls) == 2
    ranks = rankings.poll_ranks(polls, rankings.COMMITTEE_POLL)
    assert ranks.to_dict() == {"Oregon": 1.0, "Georgia": 2.0}
    assert rankings.poll_ranks(polls, rankings.AP_POLL).empty


def test_ranking_tabs_layout():
    mean = ratings({"A": 5, "B": 0, "C": -5, "D": 2})
    games = pd.DataFrame([game("A", "B", 7), game("C", "D", -3), game("A", "D", 1)])
    conferences = {"A": "X", "B": "X", "C": "Y", "D": simulate.INDEPENDENTS}
    table = rankings.rankings_table(games, mean, np.eye(5) * 0.05, 14.0, 12.0, conferences)
    table = rankings.add_polls(table, pd.DataFrame())

    tabs = rankings.ranking_tabs(table)

    assert list(tabs) == ["Résumé", "Power", "X", "Y", simulate.INDEPENDENTS]
    assert list(tabs["Power"]["Team"]) == ["A", "D", "B", "C"]
    assert tabs["Résumé"]["Résumé rank"].is_monotonic_increasing
    assert "Conference" not in tabs["X"].columns
