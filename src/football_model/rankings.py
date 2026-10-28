"""Power and résumé rankings.

Two different questions, two rankings:

- Power: how good is a team? Sort by the ridge rating (points better than an average
  FBS team on a neutral field). Each rating has a standard error from the ridge
  posterior, Cov(b) = sigma_e^2 (X'X + alpha I)^-1 (see simulate.py), so the table shows
  a 90% interval: ranks whose intervals overlap heavily are not really separated.

- Résumé (strength of record, SOR): how impressive is what a team has done? Take a
  benchmark team, rated as the average of the power top 25, and play it through this
  team's exact schedule (same opponents, same venues). Its number of wins is a sum of
  independent Bernoulli trials with different p's, a Poisson-binomial distribution,
  computed exactly by convolving one game at a time (win_distribution). SOR is

      P(benchmark wins >= this team's wins)

  Lower is more impressive: an 11-1 record against a hard schedule is something a
  top-25 team rarely matches. Records are judged against opponents' ratings as they
  stand today. That is the definition of SOR, not leakage: a ranking describes games
  already played and predicts none of them.

Polls (AP, committee) are a benchmark for these rankings, never a model input.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from football_model import config, data, predict, spread

FCS_TEAM = spread.FCS_TEAM
HOME_FIELD = spread.HOME_FIELD
BENCHMARK_TOP_N = 25
INTERVAL_Z = 1.645  # 90% normal interval
COMMITTEE_POLL = data.COMMITTEE_POLL
AP_POLL = data.AP_POLL


def win_distribution(probs: np.ndarray) -> np.ndarray:
    """Exact distribution of the number of wins in independent games (Poisson-binomial).

    Start with P(0 wins) = 1 and add one game at a time: after a game won with
    probability p, P(k wins) = p * P_before(k - 1) + (1 - p) * P_before(k).

    Args:
        probs: Win probability of each game.

    Returns:
        Array of length len(probs) + 1: entry k is P(exactly k wins).
    """
    dist = np.array([1.0])
    for p in probs:
        dist = np.append(dist, 0.0) * (1 - p) + np.insert(dist, 0, 0.0) * p
    return dist


def tail_probability(dist: np.ndarray, wins: int) -> float:
    """P(at least `wins` wins) from a win_distribution."""
    return float(dist[max(wins, 0) :].sum())


def benchmark_rating(mean: pd.Series, fbs_teams: list[str], top_n: int = BENCHMARK_TOP_N) -> float:
    """Rating of the benchmark team: the average rating of the power top `top_n`."""
    ratings = mean[fbs_teams].sort_values(ascending=False)
    return float(ratings.head(top_n).mean())


def team_games(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """One team's games from its own side: opponent, venue, and (if played) won.

    venue is +1 at home, -1 away, 0 at a neutral site (the sign on home field). `won`
    is included only when `games` has a margin column (completed games).
    """
    home = games[games["home_team"] == team]
    away = games[games["away_team"] == team]
    neutral_h = home["neutral_site"].fillna(False).astype(bool)
    neutral_a = away["neutral_site"].fillna(False).astype(bool)
    home_side = pd.DataFrame({"opponent": home["away_team"], "venue": np.where(neutral_h, 0, 1)})
    away_side = pd.DataFrame({"opponent": away["home_team"], "venue": np.where(neutral_a, 0, -1)})
    if "margin" in games.columns:
        home_side["won"] = home["margin"] > 0
        away_side["won"] = away["margin"] < 0
    return pd.concat([home_side, away_side], ignore_index=True)


def schedule_tails(
    games: pd.DataFrame,
    teams: list[str],
    mean: pd.Series,
    sigma: float,
    benchmark: float,
    max_wins: int,
) -> np.ndarray:
    """P(benchmark wins >= w) on each team's full schedule, for w = 0..max_wins.

    Used by the season simulation: the schedule is known, so the benchmark's win
    distribution is fixed, and a simulated record's SOR is a lookup in this table.

    Args:
        games: Every game on the schedule (played or not); only opponent and venue
            are used.
        teams: Teams to compute, in row order.
        mean: Ridge ratings, including home field.
        sigma: Residual SD for win probabilities.
        benchmark: benchmark_rating output.
        max_wins: Largest win count to look up.

    Returns:
        Array (teams x max_wins + 1).
    """
    tails = np.zeros((len(teams), max_wins + 1))
    for i, team in enumerate(teams):
        played = team_games(games, team)
        dist = win_distribution(
            benchmark_win_probs(played["opponent"], played["venue"], mean, benchmark, sigma)
        )
        tail = dist[::-1].cumsum()[::-1]  # tail[w] = P(W >= w)
        n = min(len(tail), max_wins + 1)
        tails[i, :n] = tail[:n]
    return tails


def benchmark_win_probs(
    opponents: pd.Series, venue: pd.Series, mean: pd.Series, benchmark: float, sigma: float
) -> np.ndarray:
    """Benchmark team's win probability in each game: Phi((B - r_opp + venue * hfa) / sigma).

    The same normal conversion as the live win probabilities (win_prob.py).
    """
    expected = benchmark - mean[opponents].to_numpy() + venue.to_numpy() * mean[HOME_FIELD]
    return norm.cdf(expected / sigma)


def resume_table(
    games: pd.DataFrame,
    mean: pd.Series,
    sigma: float,
    conferences: dict[str, str],
    top_n: int = BENCHMARK_TOP_N,
) -> pd.DataFrame:
    """Strength of record for every FBS team, from completed games.

    Args:
        games: Completed games (clean_games output) to judge; pass only games played.
        mean: Ridge ratings (simulate.season_state mean), including home field and FCS.
        sigma: Residual SD for turning a spread into a probability (win_prob.current_sigma).
        conferences: FBS team -> conference (simulate.team_conferences).
        top_n: The benchmark is the average of the power top `top_n`.

    Returns:
        One row per FBS team, best résumé first: team, conference, wins, losses, sor
        (P(benchmark matches or beats the record)), benchmark_wins (its expected wins on
        this schedule), wins_above_benchmark, schedule_strength (the benchmark's
        expected win % on this schedule: lower is harder), resume_rank.
    """
    teams = sorted(conferences)
    benchmark = benchmark_rating(mean, teams, top_n)
    rows = []
    for team in teams:
        played = team_games(games, team)
        probs = benchmark_win_probs(played["opponent"], played["venue"], mean, benchmark, sigma)
        wins = int(played["won"].sum())
        rows.append(
            {
                "team": team,
                "conference": conferences[team],
                "wins": wins,
                "losses": len(played) - wins,
                "sor": tail_probability(win_distribution(probs), wins),
                "benchmark_wins": float(probs.sum()),
                "wins_above_benchmark": wins - float(probs.sum()),
                "schedule_strength": float(probs.mean()) if len(probs) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(["sor", "wins_above_benchmark"], ascending=[True, False])
    return table.assign(resume_rank=range(1, len(table) + 1)).reset_index(drop=True)


def power_table(
    mean: pd.Series, unscaled_cov: np.ndarray, sigma_e: float, conferences: dict[str, str]
) -> pd.DataFrame:
    """Power rating for every FBS team, with a 90% interval from the ridge posterior.

    Args:
        mean, unscaled_cov: simulate.ridge_posterior / season_state output.
        sigma_e: Per-game noise SD (simulate.split_noise); the rating's standard error
            is sigma_e * sqrt(U_ii).
        conferences: FBS team -> conference.

    Returns:
        One row per FBS team, best first: team, rating, rating_se, rating_low,
        rating_high, power_rank.
    """
    teams = sorted(conferences)
    position = [mean.index.get_loc(t) for t in teams]
    se = sigma_e * np.sqrt(np.diag(unscaled_cov)[position])
    rating = mean[teams].to_numpy()
    table = pd.DataFrame(
        {
            "team": teams,
            "rating": rating,
            "rating_se": se,
            "rating_low": rating - INTERVAL_Z * se,
            "rating_high": rating + INTERVAL_Z * se,
        }
    ).sort_values("rating", ascending=False)
    return table.assign(power_rank=range(1, len(table) + 1)).reset_index(drop=True)


def record_rank(resume: pd.DataFrame, power: pd.DataFrame) -> pd.Series:
    """The season simulation's original playoff proxy: fewest losses, then rating.

    Returns:
        Rank (1 = best) indexed by team.
    """
    both = resume[["team", "losses"]].merge(power[["team", "rating"]], on="team")
    both = both.sort_values(["losses", "rating"], ascending=[True, False])
    return pd.Series(range(1, len(both) + 1), index=both["team"].to_numpy())


def rankings_table(
    games: pd.DataFrame,
    mean: pd.Series,
    unscaled_cov: np.ndarray,
    sigma: float,
    sigma_e: float,
    conferences: dict[str, str],
) -> pd.DataFrame:
    """Résumé and power rankings side by side, best résumé first.

    Args:
        games: Completed games to judge.
        mean, unscaled_cov: Ridge posterior (simulate.season_state).
        sigma: Out-of-sample residual SD (for the résumé's win probabilities).
        sigma_e: Per-game noise SD (for the rating intervals).
        conferences: FBS team -> conference.

    Returns:
        resume_table columns plus rating, rating_se, rating_low, rating_high,
        power_rank, record_rank.
    """
    resume = resume_table(games, mean, sigma, conferences)
    power = power_table(mean, unscaled_cov, sigma_e, conferences)
    table = resume.merge(power, on="team")
    return table.assign(record_rank=table["team"].map(record_rank(resume, power)))


# --- Polls (benchmark only) ---------------------------------------------------------


def poll_ranks(polls: pd.DataFrame, poll: str, week: int | None = None) -> pd.Series:
    """One poll's ranks (1 = best) indexed by school, from data.load_rankings output.

    Args:
        polls: Raw poll rows (data.load_rankings / load_rankings_week).
        poll: Poll name, e.g. COMMITTEE_POLL or AP_POLL.
        week: Poll week; None means the last regular-season week of that poll.

    Returns:
        Rank by school; empty if the poll isn't there.
    """
    if polls.empty:
        return pd.Series(dtype=float)
    rows = polls[(polls["poll"] == poll) & (polls["seasonType"] == "regular")]
    if rows.empty:
        return pd.Series(dtype=float)
    week = rows["week"].max() if week is None else week
    rows = rows[rows["week"] == week]
    return rows.set_index("school")["rank"].astype(float)


def compare_to_poll(ranks: pd.Series, poll: pd.Series) -> dict[str, float]:
    """How closely a ranking matches a poll, on the poll's ranked teams.

    Args:
        ranks: A ranking (1 = best) indexed by team, covering every FBS team.
        poll: The poll's ranks indexed by team (usually its top 25).

    Returns:
        spearman: rank correlation on the poll's teams (Pearson correlation of the two
            sets of ranks, the nonparametric analogue of a correlation coefficient);
        top4, top12: share of the poll's top 4 / top 12 that the ranking also puts in
            its top 4 / top 12;
        mean_abs_rank_diff: average |rank - poll rank| on the poll's teams.
    """
    shared = poll.index.intersection(ranks.index)
    ours, theirs = ranks[shared].astype(float), poll[shared].astype(float)
    result = {"spearman": float(np.corrcoef(ours.rank(), theirs.rank())[0, 1])}
    for k in (4, 12):
        poll_top = set(poll[poll <= k].index)
        our_top = set(ranks[ranks <= k].index)
        result[f"top{k}"] = len(poll_top & our_top) / len(poll_top) if poll_top else np.nan
    result["mean_abs_rank_diff"] = float(np.mean(np.abs(ours - theirs)))
    return result


# --- Output -------------------------------------------------------------------------

RANKING_FORMATS = {
    "SOR": "0.0%",
    "Schedule strength": "0%",
    "Wins above benchmark": "+0.0;-0.0;0.0",
    "Rating": "+0.0;-0.0;0.0",
    "Rating 90% low": "+0.0;-0.0;0.0",
    "Rating 90% high": "+0.0;-0.0;0.0",
}


def rankings_path(season: int, after_week: int, suffix: str = ".csv") -> Path:
    """predictions/{season}/rankings_after_week_{WW}{suffix}: a dated snapshot."""
    name = f"rankings_after_week_{after_week:02d}{suffix}"
    return config.PREDICTIONS_DIR / str(season) / name


def add_polls(table: pd.DataFrame, polls: pd.DataFrame) -> pd.DataFrame:
    """Add ap_rank and committee_rank (NaN = unranked or no poll yet) from one poll week."""
    return table.assign(
        ap_rank=table["team"].map(poll_ranks(polls, AP_POLL)),
        committee_rank=table["team"].map(poll_ranks(polls, COMMITTEE_POLL)),
    )


def readable_rankings(table: pd.DataFrame) -> pd.DataFrame:
    """Display columns for a rankings table (rankings_table + add_polls), best résumé first."""
    ordered = table.sort_values("resume_rank")
    return pd.DataFrame(
        {
            "Résumé rank": ordered["resume_rank"],
            "Team": ordered["team"],
            "Conference": ordered["conference"],
            "Record": [f"{w}-{l}" for w, l in zip(ordered["wins"], ordered["losses"])],
            "SOR": ordered["sor"],
            "Wins above benchmark": ordered["wins_above_benchmark"],
            "Schedule strength": ordered["schedule_strength"],
            "Power rank": ordered["power_rank"],
            "Rating": ordered["rating"],
            "Rating 90% low": ordered["rating_low"],
            "Rating 90% high": ordered["rating_high"],
            "AP": ordered.get("ap_rank"),
            "Committee": ordered.get("committee_rank"),
        }
    ).reset_index(drop=True)


def ranking_tabs(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tabs: "Résumé" (all teams), "Power" (all teams, by rating), then one per conference.

    Conference tabs are in résumé order and leave out the conference column.
    """
    readable = readable_rankings(table)
    power = readable.sort_values("Power rank")
    power = power[["Power rank", *[c for c in readable.columns if c != "Power rank"]]]
    tabs = {"Résumé": readable, "Power": power.reset_index(drop=True)}
    independents = "FBS Independents"
    for conference in sorted(readable["Conference"].unique(), key=lambda c: (c == independents, c)):
        rows = readable[readable["Conference"] == conference].drop(columns="Conference")
        # Excel sheet names: at most 31 characters, no / \ ? * [ ] :
        name = "".join(ch for ch in conference if ch not in "/\\?*[]:")[:31]
        tabs[name] = rows.reset_index(drop=True)
    return tabs


def write_rankings_excel(table: pd.DataFrame, path: Path) -> None:
    """Excel copy of the rankings, styled like the other workbooks (the CSV is the record).

    SOR (the résumé measure) is highlighted.
    """
    predict.write_tabs(
        ranking_tabs(table),
        path,
        percent_columns=[],
        number_columns=[],
        freeze="C2",
        highlight_columns=["SOR"],
        formats=RANKING_FORMATS,
    )
