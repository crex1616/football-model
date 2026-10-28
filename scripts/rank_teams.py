"""Power and résumé rankings for the current season, after a finished week.

Usage:
    python scripts/rank_teams.py --week 8 --dry-run   # print only
    python scripts/rank_teams.py --week 8             # save the snapshot

Uses every completed game of weeks 1 through --week (cached once final, so no API calls
for finished weeks) plus that week's polls for comparison (about 1 call). Saves
predictions/YYYY/rankings_after_week_WW.csv (commit it) and a readable .xlsx copy
("Résumé" and "Power" tabs, then one per conference). A saved CSV is never overwritten.
"""

import argparse

import pandas as pd

from football_model import clean, config, data, rankings, simulate, spread, win_prob


def main() -> None:
    """Build the rankings, print the top 25s and poll disagreements, save the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True, help="Last finished week.")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving.")
    args = parser.parse_args()
    season = config.CURRENT_SEASON

    raw = [data.load_games_week(season, w) for w in range(1, args.week + 1)]
    raw = pd.concat([r for r in raw if not r.empty], ignore_index=True)
    played = clean.clean_games(raw)
    past = clean.load_games_table()
    past = past[past["season"] < season]
    history = pd.concat([past, played], ignore_index=True)

    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    mean, ucov = simulate.season_state(history, teams, season)
    sigma = win_prob.current_sigma(spread.run_spread(past, simulate.RIDGE), "spread_pred", season)
    sigma_e = simulate.split_noise(sigma, ucov, simulate.game_contrasts(played, mean.index))
    conferences = simulate.team_conferences(played)
    table = rankings.rankings_table(played, mean, ucov, sigma, sigma_e, conferences)
    table = rankings.add_polls(table, data.load_rankings_week(season, args.week + 1))

    print(f"\n{season}, after week {args.week}: {len(played)} games played")
    readable = rankings.readable_rankings(table)
    shown = readable.copy()
    shown["SOR"] = (shown["SOR"] * 100).round(1)
    shown["Schedule strength"] = (shown["Schedule strength"] * 100).round(0)
    for col in ["Wins above benchmark", "Rating", "Rating 90% low", "Rating 90% high"]:
        shown[col] = shown[col].round(1)
    columns = [
        "Résumé rank", "Team", "Record", "SOR", "Wins above benchmark",
        "Schedule strength", "Power rank", "Rating", "AP", "Committee",
    ]  # fmt: skip
    with pd.option_context("display.width", 170):
        print("\nRésumé top 25 (SOR % = chance an average top-25 team matches the record):")
        print(shown[columns].head(25).to_string(index=False))
        print("\nPower top 25 (rating = points better than an average FBS team, neutral field):")
        power = shown.sort_values("Power rank")
        print(
            power[["Power rank", "Team", "Record", "Rating", "Rating 90% low", "Rating 90% high"]]
            .head(25)
            .to_string(index=False)
        )

    poll = "Committee" if table["committee_rank"].notna().any() else "AP"
    ranked = shown[shown[poll].notna()].assign(gap=lambda d: d["Résumé rank"] - d[poll])
    if len(ranked):
        print(f"\nBiggest disagreements with the {poll} top 25 (résumé rank - {poll} rank):")
        biggest = ranked.reindex(ranked["gap"].abs().sort_values(ascending=False).index)
        print(
            biggest[["Team", "Record", "Résumé rank", poll, "Power rank"]]
            .head(8)
            .to_string(index=False)
        )

    if args.dry_run:
        print("\nDry run: nothing saved.")
        return
    csv = rankings.rankings_path(season, args.week)
    xlsx = rankings.rankings_path(season, args.week, ".xlsx")
    if csv.exists():
        # A saved snapshot is on record: never overwrite it. Rebuild the .xlsx only if
        # this run reproduces the saved numbers.
        if simulate.same_odds(pd.read_csv(csv), table):
            rankings.write_rankings_excel(table, xlsx)
            print(f"\n{csv.name} is already saved and this run matches it. Rebuilt {xlsx}.")
        else:
            print(f"\n{csv.name} is already saved and this run differs from it. Nothing saved.")
        return
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv, index=False)
    rankings.write_rankings_excel(table, xlsx)
    print(f"\nSaved {csv} (commit it) and the .xlsx copy.")


if __name__ == "__main__":
    main()
