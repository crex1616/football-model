"""Rebuild the readable Excel copy of a saved week of predictions.

Usage:
    python scripts/export_predictions.py --week 9

Reads predictions/YYYY/week_WW.csv (never changes it) and writes week_WW.xlsx next to
it: an "All games" tab plus one tab per conference. Files saved before conferences
were added to the CSV get them from the cached raw games for that week (an API call
only if the week is still in progress).
"""

import argparse

from football_model import config, data, predict


def main() -> None:
    """Write the Excel copy for one saved week."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True, help="Week to export.")
    parser.add_argument("--season", type=int, default=config.CURRENT_SEASON)
    args = parser.parse_args()

    preds = predict.load_saved_predictions(predict.prediction_path(args.season, args.week))
    if "home_conference" not in preds.columns:
        preds = predict.add_conferences(preds, data.load_games_week(args.season, args.week))

    path = predict.excel_path(args.season, args.week)
    predict.write_excel(preds, path)
    print(f"Saved {len(preds)} games to {path}")


if __name__ == "__main__":
    main()
