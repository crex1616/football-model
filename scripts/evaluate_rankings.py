"""Score the rankings against the CFP committee's final ranking, season by season.

Usage:
    python scripts/evaluate_rankings.py          # tuning seasons 2015-2024 excl. 2020
    python scripts/evaluate_rankings.py --test   # also the 2025 test season (rarely)

For each season, the rankings are built as of selection day: every regular-season game
through the conference title games, no bowls. Each ranking is compared to the
committee's final top 25 (data.load_rankings; 1 API call per season the first time):

- record: fewest losses, then rating (the season simulation's original playoff proxy)
- power: ridge rating
- resume: strength of record
- AP poll (same week): a human benchmark, the way the market line is for spreads.

Metrics: Spearman rank correlation on the committee's 25 teams, share of the
committee's top 4 and top 12 also in the ranking's top 4 / top 12, and mean absolute
rank difference.
"""

import argparse

import pandas as pd

from football_model import clean, config, data, rankings, simulate, spread, win_prob

RULES = ["record", "power", "resume", "ap"]


def season_rankings(
    games: pd.DataFrame, season: int, fits: dict[int, pd.Series], sigma: float
) -> pd.DataFrame:
    """Rankings as of selection day (regular season incl. title games) for one season."""
    played = games[(games["season"] == season) & (games["season_type"] == "regular")]
    history = games[(games["season"] < season) | games["game_id"].isin(played["game_id"])]
    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    mean, ucov = simulate.season_state(history, teams, season, fits=fits)
    sigma_e = simulate.split_noise(sigma, ucov, simulate.game_contrasts(played, mean.index))
    conferences = simulate.team_conferences(played)
    return rankings.rankings_table(played, mean, ucov, sigma, sigma_e, conferences)


def score_season(table: pd.DataFrame, polls: pd.DataFrame) -> pd.DataFrame:
    """One row per rule: how closely it matches the committee's final ranking."""
    committee = rankings.poll_ranks(polls, rankings.COMMITTEE_POLL)
    week = polls.loc[polls["poll"] == rankings.COMMITTEE_POLL, "week"]
    week = week[polls["seasonType"] == "regular"].max()
    ap = rankings.poll_ranks(polls, rankings.AP_POLL, week=week)
    by_team = table.set_index("team")
    ranks = {
        "record": by_team["record_rank"],
        "power": by_team["power_rank"],
        "resume": by_team["resume_rank"],
        # The AP ranks only 25 teams; unranked teams share rank 26 for the comparison.
        "ap": ap.reindex(by_team.index).fillna(26.0),
    }
    rows = [{"rule": r, **rankings.compare_to_poll(ranks[r], committee)} for r in RULES]
    return pd.DataFrame(rows)


def main() -> None:
    """Score every season and print per-season and average results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON].reset_index(drop=True)
    params = simulate.RIDGE
    fits = spread.season_fits(games, params)
    walk_forward = spread.run_spread(games, params)

    seasons = list(config.TUNING_SEASONS)
    if args.test:
        seasons.append(config.TEST_SEASON)
    parts, tables = [], {}
    for season in seasons:
        sigma = win_prob.current_sigma(walk_forward, "spread_pred", season)
        table = season_rankings(games, season, fits, sigma)
        tables[season] = table
        parts.append(score_season(table, data.load_rankings(season)).assign(season=season))
    scores = pd.concat(parts, ignore_index=True)

    pd.set_option("display.width", 160)
    metrics = ["spearman", "top4", "top12", "mean_abs_rank_diff"]
    tuning = scores[scores["season"].isin(config.TUNING_SEASONS)]
    print("\nPer season, Spearman with the committee's final top 25:")
    print(tuning.pivot(index="season", columns="rule", values="spearman")[RULES].round(3))
    print(f"\nTuning seasons ({tuning['season'].nunique()}), averages:")
    print(tuning.groupby("rule")[metrics].mean().loc[RULES].round(3))
    if args.test:
        test = scores[scores["season"] == config.TEST_SEASON]
        print(f"\nTEST season {config.TEST_SEASON}:")
        print(test.set_index("rule")[metrics].loc[RULES].round(3))

    last = max(s for s in config.TUNING_SEASONS)
    committee = rankings.poll_ranks(data.load_rankings(last), rankings.COMMITTEE_POLL)
    show = tables[last].assign(committee=tables[last]["team"].map(committee))
    show = show[show["resume_rank"] <= 15]
    print(f"\n{last} résumé top 15 vs. committee:")
    columns = ["resume_rank", "team", "wins", "losses", "sor", "power_rank", "committee"]
    print(show[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
