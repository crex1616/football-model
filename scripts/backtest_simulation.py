"""Backtest the season simulation: simulate past seasons from week 10, compare to reality.

Usage:
    python scripts/backtest_simulation.py            # tuning seasons 2016-2024 excl. 2020
    python scripts/backtest_simulation.py --test     # also the 2025 test season (rarely)

For each season, the simulation starts from what was known before week 10's first
kickoff: ratings fit on earlier games, and the rest of the regular-season schedule
(past title games are removed, since their matchups weren't known yet). Scores:

- Expected regular-season wins: MAE vs. actual, and vs. a naive baseline that assumes
  each team keeps its current win rate.
- P(bowl eligible): log loss, Brier, calibration plot.
- P(conference champion): log loss and Brier for 2024 on, the first season the team
  table's divisions are right (before that, most conferences had divisions this
  project doesn't have history for).

No API calls (reads the games table and the cached team list).
"""

import argparse

import numpy as np
import pandas as pd

from football_model import clean, config, data, evaluate, simulate, spread, win_prob

N_SIMS = 2_000
START_WEEK = 10
FIRST_SEASON_WITH_DIVISIONS_KNOWN = 2024


def actual_outcomes(season_games: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Final regular-season wins, bowl eligibility, and conference champion per team."""
    regular = season_games[season_games["season_type"] == "regular"]
    title = simulate.is_title_game(regular)
    reg_w, _, _ = simulate._completed_records(regular[~title], teams)
    all_w, _, fcs_w = simulate._completed_records(regular, teams)
    champs = set()
    for _, g in regular[title].iterrows():
        champs.add(g["home_team"] if g["margin"] > 0 else g["away_team"])
    return pd.DataFrame(
        {
            "team": teams,
            "actual_regular_wins": reg_w,
            "actual_bowl": ((all_w - np.maximum(fcs_w - 1, 0)) >= simulate.BOWL_WINS).astype(int),
            "actual_champ": [int(t in champs) for t in teams],
        }
    )


def backtest_season(
    games: pd.DataFrame,
    season: int,
    fits: dict[int, pd.Series],
    sigma_oos: float,
    divisions: dict[str, str],
) -> pd.DataFrame:
    """Simulate one past season from week START_WEEK and join the actual outcomes."""
    season_games = games[games["season"] == season]
    regular = season_games[season_games["season_type"] == "regular"]
    cutoff = regular.loc[regular["week"] == START_WEEK, "start_date"].min()

    completed = regular[regular["start_date"] < cutoff]
    remaining = regular[regular["start_date"] >= cutoff]
    remaining = remaining[~simulate.is_title_game(remaining)]
    remaining = remaining.drop(columns=["home_points", "away_points", "margin", "home_win"])

    history = games[(games["season"] < season) | games["game_id"].isin(completed["game_id"])]
    teams = sorted(set(regular["home_team"]) | set(regular["away_team"]))
    mean, ucov = simulate.season_state(history, teams, season, fits=fits)

    conferences = simulate.team_conferences(regular)
    use_divisions = divisions if season >= FIRST_SEASON_WITH_DIVISIONS_KNOWN else {}
    odds = simulate.simulate_season(
        completed,
        remaining,
        mean,
        ucov,
        sigma_oos,
        conferences,
        use_divisions,
        n_sims=N_SIMS,
        seed=config.RANDOM_SEED,
    )

    # Naive baseline: current wins + current win rate x remaining games.
    fbs = sorted(conferences)
    played = odds["wins"] + odds["losses"]
    home_inc, away_inc = simulate._incidence(remaining, fbs)
    left = (home_inc + away_inc).sum(axis=1)
    rate = np.divide(odds["wins"], played, out=np.full(len(odds), 0.5), where=played > 0)
    odds["naive_regular_wins"] = odds["wins"] + rate * left
    odds["season"] = season
    return odds.merge(actual_outcomes(season_games, fbs), on="team")


def summarize(results: pd.DataFrame, label: str) -> None:
    """Print the three scores for a set of backtested seasons."""
    wins_mae = evaluate.mae(results["actual_regular_wins"], results["exp_regular_wins"])
    naive_mae = evaluate.mae(results["actual_regular_wins"], results["naive_regular_wins"])
    print(f"\n{label}: {results['season'].nunique()} season(s), {len(results)} team-seasons")
    print(f"  Regular-season wins MAE: simulation {wins_mae:.3f}, naive {naive_mae:.3f}")

    p = results["p_bowl"].clip(1e-6, 1 - 1e-6)
    print(
        f"  Bowl eligible: log loss {evaluate.log_loss(results['actual_bowl'], p):.4f}, "
        f"Brier {evaluate.brier_score(results['actual_bowl'], p):.4f} "
        f"(base rate {results['actual_bowl'].mean():.3f})"
    )

    champ = results[results["season"] >= FIRST_SEASON_WITH_DIVISIONS_KNOWN]
    champ = champ[champ["conference"] != simulate.INDEPENDENTS]
    if len(champ) and champ["actual_champ"].sum():
        p = champ["p_conf_champ"].clip(1e-6, 1 - 1e-6)
        print(
            f"  Conference champion ({sorted(champ['season'].unique())}): "
            f"log loss {evaluate.log_loss(champ['actual_champ'], p):.4f}, "
            f"Brier {evaluate.brier_score(champ['actual_champ'], p):.4f}; "
            f"champions' average probability {champ.loc[champ['actual_champ'] == 1, 'p_conf_champ'].mean():.3f}"
        )


def main() -> None:
    """Backtest every scored season and print the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also score the test season.")
    args = parser.parse_args()

    games = clean.load_games_table()
    games = games[games["season"] <= config.TEST_SEASON].reset_index(drop=True)
    params = spread.SpreadParams()
    fits = spread.season_fits(games, params)
    walk_forward = spread.run_spread(games, params)

    teams = data.load_teams()
    divisions = dict(zip(teams["school"], teams["division"].where(teams["division"].notna())))
    divisions = {t: d for t, d in divisions.items() if isinstance(d, str)}

    seasons = [s for s in config.TUNING_SEASONS if s >= 2016]
    if args.test:
        seasons.append(config.TEST_SEASON)
    parts = []
    for season in seasons:
        sigma = win_prob.current_sigma(walk_forward, "spread_pred", season)
        parts.append(backtest_season(games, season, fits, sigma, divisions))
        print(f"  simulated {season}")
    results = pd.concat(parts, ignore_index=True)

    tuning = results[results["season"].isin(config.TUNING_SEASONS)]
    summarize(tuning, "Tuning seasons")
    path = config.FIGURES_DIR / "simulation_bowl_calibration.png"
    evaluate.calibration_plot(
        tuning["actual_bowl"],
        tuning["p_bowl"],
        path,
        title=f"P(bowl eligible) from week {START_WEEK}, tuning seasons",
        outcome="bowl eligible",
    )
    print(f"  Saved calibration plot to {path}")
    if args.test:
        summarize(results[results["season"] == config.TEST_SEASON], "TEST season")


if __name__ == "__main__":
    main()
