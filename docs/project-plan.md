# College Football Prediction Model: Project Plan

Snapshot of the plan as of Oct 1, 2026. The live, editable version is the [Claude doc](https://claude.ai/code/artifact/ea42c5ea-9b79-42b6-9fdb-ed25dc109b00).

## Goals and outputs

The goal is a college football model that updates every week and produces four outputs, while teaching core machine-learning practice along the way.

- **Predicted point spread** for every game (a linear regression on point margin).
- **Win probability** for every game, derived from the spread or fit directly with logistic regression.
- **Season outcome probabilities**: expected wins, conference title odds, and playoff odds, from Monte Carlo simulation.
- **Team rankings** built from the model's power ratings.

Success means calibrated probabilities on games the model has never seen, measured honestly against the betting market. It does not mean picking the most winners.

## From econometrics to machine learning

Your regression background covers most of the math. The real change is the objective: prediction on unseen data instead of inference about coefficients.

| | Econometrics habit | Prediction habit |
| --- | --- | --- |
| Goal | Unbiased, interpretable coefficients | Accurate probabilities on future games |
| Model judged by | Standard errors, in-sample fit | Out-of-sample log loss and calibration |
| More variables | Control for confounders | Often hurts: overfitting |
| Multicollinearity | A problem for inference | Mostly harmless |
| Tuning | Theory-driven specification | Regularization strength chosen by validation |

Three ML ideas to learn early:

- **Train / validation / test splits.** Fit on one block of data, tune on a second, and report results on a third you touch once.
- **Regularization.** Ridge and lasso shrink coefficients toward zero to trade a little bias for much less variance.
- **Hyperparameters.** Settings such as the Elo update speed or ridge penalty that you pick by validation, not by estimation.

## Tools and setup

| Tool | What it's for |
| --- | --- |
| Python + Jupyter (or VS Code) | Writing and exploring code interactively |
| pandas | Loading, cleaning, and joining game data |
| statsmodels | Regressions with familiar econometric output |
| scikit-learn | Logistic/ridge models, cross-validation, metrics |
| LightGBM or XGBoost | Gradient boosting, for the later model comparison |
| matplotlib | Calibration plots and rating charts |
| Git + GitHub | Version control so you can track model changes |

## Getting the data

CollegeFootballData.com (CFBD) is the main source: one free API key gives you games, scores, team stats, recruiting, advanced metrics, and historical betting lines ([tiers](https://collegefootballdata.com/api-tiers)).

| Source | What you get | Cost / limits |
| --- | --- | --- |
| [CFBD API](https://collegefootballdata.com/key), free tier | Games, scores, team and player stats, recruiting, betting lines, EPA/PPA | Free, 1,000 calls/month |
| CFBD Academic tier | Same data | Free with a .edu email, 3,000 calls/month |
| CFBD Tier 1 / Tier 2 (Patreon) | Adds opponent-adjusted metrics, weather, live scores; Tier 2 adds live play-by-play | $1/month (5,000 calls) / $5/month (30,000 calls) |
| [cfbd-python](https://github.com/CFBD/cfbd-python) | Official Python client for the API | Free |
| [CFBD Data Exporter](https://collegefootballdata.com/exporter) | Point-and-click CSV downloads to test queries | Free |

**Stay under the free limit by caching.** Pull each historical season once, save it locally, and only request the current week afterward. Ten seasons of history costs a few dozen calls.

**What to pull first:**

1. Games with final scores, home/away, neutral-site flag, and conference, 2014 to present.
2. Betting lines (closing spread and over/under) for the same games, as your benchmark.
3. Team season stats and advanced metrics (EPA per play, success rate), for later features.
4. Recruiting rankings and returning production, for preseason ratings.

Note: the 2020 season was heavily disrupted by COVID. Consider down-weighting or excluding it.

## Build phases

Build in seven phases, each a working model on its own. Every later phase must beat the one before it on out-of-sample log loss to earn its place.

```
Game data ──> Team ratings ──> Predicted spread ──> Win probability
   ^           (Elo, ridge)                              │
   │                                                     v
   └── new results each week ── Rankings & odds <── Season simulation
```

### Phase 1: Data pipeline

- Write one script that pulls each season from CFBD and saves it locally.
- Build a clean game table: date, season, week, home team, away team, scores, neutral site, conference game flag, closing spread.
- Lump all FCS teams into one shared "FCS" team to start.
- Add `margin = home_points - away_points` and `home_win = 1 if margin > 0`.

### Phase 2: Elo baseline

Each team has a rating; after each game the winner gains points and the loser loses the same amount.

```
E_home = 1 / (1 + 10^(-(R_home - R_away + H) / 400))
R_home_new = R_home + K * M * (S_home - E_home)
```

S is 1 for a win and 0 for a loss, H is home-field advantage in Elo points, K sets how fast ratings move, and M is a margin-of-victory multiplier. Tune K, H, and M on past seasons. Between seasons, pull every rating partway back toward the mean.

### Phase 3: Point-spread regression

Regress home margin on team indicator variables: +1 for the home team, −1 for the away team, 0 for everyone else.

```
margin_g = h * home_g + Σ_t β_t x_{g,t} + ε_g
```

Each β is a team's rating in points versus an average team, and h is home-field advantage. The predicted spread is h + β_home − β_away.

- **Use ridge regression** so thin early-season data doesn't produce wild ratings.
- **Refit every week** on games played so far; optionally weight recent games more.
- **Consider capping margins** and test whether it helps.
- **Start the season with a prior** from last season's rating.

### Phase 4: Win probabilities

```
P(home win) = Φ(predicted_margin / σ)
```

σ is the standard deviation of out-of-sample residuals. Cross-check with a logistic regression of `home_win` on the predicted spread.

### Phase 5: Richer features and ML models

Compare ridge regression, (penalized) logistic regression, and LightGBM on the same walk-forward splits. Candidate features: rating difference, opponent-adjusted EPA per play, success rate, turnover margin, rest days, travel distance, recruiting composite, returning production.

### Phase 6: Season simulation

Simulate the remaining schedule 10,000 times, applying tiebreakers and championship games. Let ratings move within each simulation. Approximate committee playoff selection with a rule, checking the current format first.

### Phase 7: Rankings

- **Power ranking:** sort by rating.
- **Résumé ranking (strength of record):** probability an average top-25 team would match this record against the same schedule.

## Validation and evaluation

- **Walk-forward validation** only; never random k-fold.
- **No leakage:** every feature knowable before kickoff; closing spreads are a benchmark, not a feature.
- **Metrics:** MAE for spreads; log loss, Brier score, calibration plots for probabilities.
- **Benchmark:** the closing market line, scored with the same metrics.

## College football challenges

- Sparse schedules (about 130 FBS teams, ~12 games each) make shrinkage essential.
- Early-season uncertainty: preseason priors carry the model until about week 5.
- Roster turnover from the transfer portal and NIL.
- Blowouts and garbage time inflate margins.
- FCS opponents have little data.
- Injuries and quarterback changes are the biggest single-game swings.
- The playoff is committee-picked, so simulations only approximate selection.

## Pacing and weekly workflow

1. **Now through mid-October:** Phases 1–2, backtested on 2014–2025.
2. **Mid-October through the regular season:** Phases 3–4, predicting each week live.
3. **Conference championships and bowls:** Phases 6–7.
4. **Offseason:** Phase 5 and a full review of the 2026 season.

Each week in season:

- [ ] Pull last week's results and next week's schedule and lines
- [ ] Refit or update ratings with the new games
- [ ] Save predictions for next week before kickoff, with a timestamp
- [ ] Score last week's predictions against results and the market
- [ ] Rerun the season simulation and update rankings
- [ ] Log one thing the model missed and a hypothesis why
