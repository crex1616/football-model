"""Simulate the rest of the current season: expected wins, bowls, conference titles, playoff.

Usage:
    python scripts/simulate_season.py               # 10,000 simulations
    python scripts/simulate_season.py --sims 2000   # quicker
    python scripts/simulate_season.py --dry-run     # print only, save nothing

Uses the live model (ridge ratings) fit on every game played so far. Saves a dated
snapshot, predictions/YYYY/season_odds_after_week_WW.csv (commit it), and a readable
.xlsx copy ("Playoff odds" tab, then one tab per conference). A saved CSV is never
overwritten: if it exists and this run reproduces it, only the .xlsx is rebuilt.
API calls: about one per week still to play (those weeks are re-fetched until final).
"""

import argparse

import pandas as pd

from football_model import clean, config, data, simulate, spread, win_prob


def main() -> None:
    """Simulate the season, print the headline odds, and save the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=10_000, help="Simulated seasons.")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving.")
    args = parser.parse_args()
    season = config.CURRENT_SEASON

    past = clean.load_games_table()
    past = past[past["season"] < season]
    completed, remaining = clean.build_full_season(season)
    history = pd.concat([past, completed], ignore_index=True)
    schedule = pd.concat([completed, remaining], ignore_index=True)

    teams = sorted(set(schedule["home_team"]) | set(schedule["away_team"]))
    mean, ucov = simulate.season_state(history, teams, season)
    walk_forward = spread.run_spread(past, simulate.RIDGE)
    sigma_oos = win_prob.current_sigma(walk_forward, "spread_pred", season)

    team_table = data.load_teams()
    divisions = {
        school: div
        for school, div in zip(team_table["school"], team_table["division"])
        if isinstance(div, str)
    }
    conferences = simulate.team_conferences(schedule)
    odds = simulate.simulate_season(
        completed,
        remaining,
        mean,
        ucov,
        sigma_oos,
        conferences,
        divisions,
        n_sims=args.sims,
        seed=config.RANDOM_SEED,
    )

    unfinished_weeks = set(remaining["week"])
    after_week = max(w for w in set(completed["week"]) if all(u > w for u in unfinished_weeks))
    print(
        f"\n{season}, after week {after_week}: {len(completed)} games played, "
        f"{len(remaining)} to play, {args.sims:,} simulations"
    )

    table = simulate.odds_table(odds)
    pct = simulate.ODDS_PERCENT_COLUMNS
    shown = table.copy()
    shown[pct] = (shown[pct] * 100).round(1)
    shown["Expected wins"] = shown["Expected wins"].round(1)
    with pd.option_context("display.width", 160):
        print("\nTop 25 by playoff odds (%):")
        print(shown.head(25).to_string(index=False))
        print("\nConference title favorites (%):")
        for conference, group in shown.groupby("Conference"):
            if conference == simulate.INDEPENDENTS:
                continue
            top = group.sort_values("Win conference", ascending=False).head(3)
            favorites = ", ".join(
                f"{t} {p:.0f}" for t, p in zip(top["Team"], top["Win conference"])
            )
            print(f"  {conference:18s} {favorites}")

    by_conf = odds[odds["conference"] != simulate.INDEPENDENTS].groupby("conference")
    print(
        f"\nChecks: title odds per conference sum to {by_conf['p_conf_champ'].sum().round(3).unique()}, "
        f"playoff odds sum to {odds['p_playoff_resume'].sum():.2f} (résumé rule) and "
        f"{odds['p_playoff'].sum():.2f} (record rule)"
    )

    if args.dry_run:
        print("\nDry run: nothing saved.")
        return
    csv = simulate.odds_path(season, after_week)
    xlsx = simulate.odds_path(season, after_week, ".xlsx")
    if csv.exists():
        # A saved snapshot is a forecast on record: never overwrite it. Rebuild the
        # .xlsx only if this run reproduces the saved numbers.
        if simulate.same_odds(pd.read_csv(csv), odds):
            simulate.write_odds_excel(odds, xlsx)
            print(f"\n{csv.name} is already saved and this run matches it. Rebuilt {xlsx}.")
        else:
            print(
                f"\n{csv.name} is already saved and this run differs from it (more games "
                "have finished). Nothing saved; rerun once the week is final."
            )
        return
    csv.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(csv, index=False)
    simulate.write_odds_excel(odds, xlsx)
    print(f"\nSaved {csv} (commit it) and the .xlsx copy.")


if __name__ == "__main__":
    main()
