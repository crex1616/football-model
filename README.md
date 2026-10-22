# football-model

A college football model that predicts point spreads and win probabilities for every FBS game, simulates the rest of the season, and ranks teams. It is also a learning project: a way to build machine-learning skills on top of an econometrics background, and to improve the model week by week against real results.

## What it produces

| Output | How it's made |
| --- | --- |
| Predicted point spread | Ridge regression of home margin on team indicators, home field, and features |
| Win probability | Normal curve on the predicted spread, checked against a logistic regression |
| Season odds | 10,000 Monte Carlo simulations of the remaining schedule (expected wins, bowl eligibility, conference and playoff odds) |
| Rankings | Power ranking (team ratings) and résumé ranking (strength of record) |

## How the model works

```
Game data ──> Team ratings ──> Predicted spread ──> Win probability
   ^           (Elo, ridge)                              │
   │                                                     v
   └── new results each week ── Rankings & odds <── Season simulation
```

The project is built in seven phases. Each phase is a working model, and each later phase has to beat the previous one on out-of-sample log loss to stay.

1. **Data pipeline** – pull and cache games, scores, and betting lines from CollegeFootballData.com
2. **Elo baseline** – a rating that updates after every game
3. **Point-spread regression** – ridge regression on point margin; team coefficients are ratings in points
4. **Win probabilities** – convert spreads to probabilities
5. **Richer features and ML models** – efficiency stats, recruiting, returning production; gradient boosting
6. **Season simulation** – Monte Carlo playoff and conference odds
7. **Rankings** – power and résumé rankings

The full plan lives in [docs/project-plan.md](docs/project-plan.md).

## How it's evaluated

- **Walk-forward validation only.** Train through season N, predict season N+1. Within a season, predict week W using only earlier weeks.
- **Metrics:** mean absolute error for spreads; log loss, Brier score, and calibration plots for probabilities.
- **Benchmark:** the closing betting line. The gap to the market is the main measure of progress.
- **Honest track record:** live predictions are saved to `predictions/` before kickoff and never edited afterward.

## Project layout

```
football-model/
├── CLAUDE.md              Conventions for Claude (and for you) when writing code here
├── README.md
├── pyproject.toml         Package definition so `import football_model` works everywhere
├── requirements.txt
├── .env.example           Template for your API key (copy to .env)
├── data/
│   ├── raw/               Untouched API pulls (not committed)
│   ├── interim/           Cleaned intermediate tables (not committed)
│   └── processed/         Model-ready tables (not committed)
├── docs/                  Project plan and notes
├── notebooks/             Exploration and learning notebooks
├── predictions/           Weekly predictions, saved before kickoff (committed)
├── reports/figures/       Calibration plots, rating charts
├── scripts/               Command-line entry points (fetch data, weekly update)
├── src/football_model/    The reusable code: data, ratings, models, simulation, evaluation
└── tests/                 Tests, especially leakage checks
```

## Setup

Uses a conda environment (Python 3.12) defined in `environment.yml`. Requires [Miniforge](https://github.com/conda-forge/miniforge) or another conda install.

```bash
git clone https://github.com/crex1616/football-model.git
cd football-model

conda env create -f environment.yml   # also installs this package in editable mode (`import football_model`)
conda activate football-model

cp .env.example .env                  # then paste your CFBD API key into .env
```

After adding a package to `environment.yml`, run `conda env update -f environment.yml --prune`. In Jupyter or VS Code, choose the **Python (football-model)** kernel.

`requirements.txt` is kept for plain-pip installs (`pip install -r requirements.txt && pip install -e .`).

Get a free API key at [collegefootballdata.com/key](https://collegefootballdata.com/key). The free tier allows 1,000 calls per month, which is plenty because every pull is cached in `data/raw/`.

## Data source

All data comes from [CollegeFootballData.com](https://collegefootballdata.com) through the official [cfbd-python](https://github.com/CFBD/cfbd-python) client. Raw data is not committed to this repository; anyone can rebuild it with their own key.

## Status

| Phase | Status |
| --- | --- |
| 1. Data pipeline | Done |
| 2. Elo baseline | Done |
| 3. Point-spread regression | Done |
| 4. Win probabilities | Done |
| 5. Features and ML models | Not started |
| 6. Season simulation | Not started |
| 7. Rankings | Not started |
