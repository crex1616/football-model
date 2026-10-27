"""Score a saved week of predictions against final results and the closing line.

Usage:
    python scripts/score_week.py --week 9

Reads predictions/YYYY/week_WW.csv (never changes it) and the week's games and lines
from data/raw/ (re-fetched only while the week is unfinished: about 2 API calls the
first run after the week is over, then none). Prints model vs. market for the week and
the season so far, plus the biggest misses. Once every game is final, writes
predictions/YYYY/week_WW_results.csv; while games remain, it only prints.
"""

import argparse

import pandas as pd

from football_model import clean, config, data, evaluate, predict


def main() -> None:
    """Score one week; write its results file once the week is complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True, help="Week to score.")
    args = parser.parse_args()
    season = config.CURRENT_SEASON

    preds = predict.load_saved_predictions(predict.prediction_path(season, args.week))
    raw = data.load_games_week(season, args.week)
    results = clean.clean_games(raw)
    n_final = int(preds["game_id"].isin(results["game_id"]).sum())
    print(f"\n{season} week {args.week}: {n_final} of {len(preds)} predicted games final.")
    if n_final == 0:
        print("Nothing to score yet.")
        return

    closing = clean.consensus_lines(data.load_lines_week(season, args.week))
    past = clean.load_games_table()
    sigma = evaluate.market_residual_sd(past, [s for s in past["season"].unique() if s < season])
    scored = predict.score_predictions(preds, results, closing, sigma)

    print("\nThis week (games with a closing line):")
    print(predict.summarize_scores(scored).round(3).to_string())
    ats = scored["ats_result"].value_counts()
    record = f"{ats.get('win', 0)}-{ats.get('loss', 0)}-{ats.get('push', 0)}"
    print(f"Ridge against the closing line: {record} (for reading only; noisy)")

    print("\nBiggest misses (ridge):")
    print(predict.biggest_misses(scored).to_string(index=False))

    path = predict.results_path(season, args.week)
    if n_final < len(preds):
        print(f"\n{len(preds) - n_final} game(s) not final; results file not written yet.")
    elif path.exists():
        print(f"\n{path.name} already exists; not rewritten.")
    else:
        scored.to_csv(path, index=False)
        print(f"\nSaved {path}")

    files = sorted(path.parent.glob("week_*_results.csv"))
    if files:
        season_scored = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        weeks = ", ".join(f.name[5:7] for f in files)
        print(f"\nSeason to date (weeks {weeks}):")
        print(predict.summarize_scores(season_scored).round(3).to_string())


if __name__ == "__main__":
    main()
