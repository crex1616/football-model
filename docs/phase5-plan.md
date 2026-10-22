# Phase 5 plan: richer features and ML models

## Goal

After Phase 4, the ridge spread model trails the market by 0.026 log loss on the tuning
seasons (0.500 vs. 0.474), and most of the gap is in **weeks 1–4** (0.415 vs. 0.374).
Phase 5 adds information the market has and the model doesn't:

- **Preseason roster strength** (recruiting, talent, returning production), aimed at
  the early-season gap.
- **Play-level efficiency** (EPA per play, success rate), which is less noisy than final
  margins.
- **Situational factors** (rest, travel).

It then compares linear models with LightGBM on the same walk-forward splits.

Phase 5 is split into three steps, each with its own commit, tests, and before/after
scores.

## 5a. Data (about 60 API calls)

Loaders in `data.py`, all through the existing `_cached` helper, so each completed
season is fetched once.

| Data | CFBD call | Calls |
|---|---|---|
| Per-game advanced stats: EPA/play, success rate, offense and defense (garbage time excluded) | `StatsApi.get_advanced_game_stats(year)` | 12 (2014–2025) |
| Team recruiting rankings | `RecruitingApi.get_team_recruiting_rankings(year)` | 17 (2010–2026; 4-year averages need 2010–13) |
| Talent composite | `TeamsApi.get_talent(year)` | 13 (2014–2026) |
| Returning production | `PlayersApi.get_returning_production(year)` | 13 (2014–2026) |
| Venue locations | `VenuesApi.get_venues()` | 1 |
| Team locations | `TeamsApi.get_teams()` | 1 |

- Rest days come from the existing games table. `clean.py` gains `venue_id`, `home_id`,
  and `away_id` (IDs keep the real school for teams mapped to "FCS").
- **Turnover margin is deferred.** Its endpoint requires a week argument (~200 calls),
  and turnovers are mostly noise. Revisit if everything else works.
- **Not allowed as features:** CFBD's SP+, FPI, and Elo ratings. They are other people's
  models, excluded like `cfbd_home_elo`.

## 5b. Features (`features.py`)

Every feature uses only information available before kickoff, and each feature function
gets a leakage test.

1. **Opponent-adjusted efficiency.** The same fixed-effects ridge as `spread.py`, on
   offensive EPA/play: `off_epa(team, game) = O_team − D_opponent + h + e`. Each team
   gets offense and defense coefficients, refit weekly (walk-forward) with last
   season's values as the prior. The same model is fit for success rate.
2. **Preseason strength:** recruiting composite (4-year average), talent, returning
   production, and last season's final rating. Each also gets a version multiplied by
   `1 / (1 + games played)`, so roster information matters in September and fades as
   games accumulate.
3. **Situational:** rest-day difference and the away team's travel distance (haversine;
   no new library).
4. **Existing ratings:** ridge spread, Huber spread, Elo difference.

Every feature is a home-minus-away difference (one row per game, home perspective).

## 5c. Models (`models.py`)

New models are added alongside the existing ones. The Phase 3/4 model stays as the
baseline.

| Model | What it is | Econometrics analogue |
|---|---|---|
| Baseline | Phase 3 ridge spread + Φ(spread/σ) | — |
| Stacked linear | Ridge regression of margin on standardized features, then Φ(·/σ) | OLS with a shrinkage penalty, with a statsmodels coefficient table |
| Penalized logistic | `home_win` on the same features | L2-penalized logit |
| LightGBM | Gradient-boosted trees on margin, then Φ(·/σ) | Flexible nonparametric regression that finds interactions; fit stage by stage, each tree fitting the previous trees' residuals |

- **Training:** to predict season N, train on seasons before N. Features are themselves
  walk-forward, so every training row looks like a real pre-game forecast.
  Hyperparameters are tuned on the tuning seasons, with seeds from `config.RANDOM_SEED`.
- **Feature groups are added one at a time** (ratings only → + efficiency → + preseason
  → + situational). Each is kept only if walk-forward log loss improves, and the
  comparison goes in the commit message.

## Decisions

- **Scoring starts at 2016.** Stacked models need at least one earlier season to train
  on; the baseline is re-scored on the same 2016–2024 seasons so the comparison is fair.
- **2020 is excluded from stacked-model training** (short, irregular, conference-only
  schedules). It is still used for features and priors.
- **New dependency:** `lightgbm`, added to `environment.yml` and `requirements.txt` in
  5c only.

## Files

- **Modified:** `data.py`, `clean.py`, `spread.py` (make `_ridge_solve` public),
  `CLAUDE.md` (module table), `README.md` (status)
- **New:** `features.py`, `models.py`, `scripts/fetch_phase5_data.py`,
  `scripts/evaluate_phase5.py`, `tests/test_features.py`, `tests/test_models.py`,
  `notebooks/05_features.ipynb`

## Verification

- `pytest` and `ruff` after each step.
- 5a: report API calls used; coverage report and team-name mismatches; spot-check a few
  teams' EPA and recruiting values against CFBD's site.
- 5b: log-loss table as each feature group is added, overall and for weeks 1–4.
- 5c: every model vs. baseline, Elo, and market, with calibration plots. Score the 2025
  test season once at the end.
