# Phase 5 plan: richer features and ML models

## Goal

After Phase 4, the ridge spread model trails the market by 0.026 log loss on the tuning
seasons (0.500 vs. 0.474), and most of the gap is in **weeks 1–4** (0.415 vs. 0.374).
Phase 5 adds information the market has and the model doesn't:

- **Preseason roster strength** (returning production), aimed at the early-season gap.
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
| Team recruiting rankings (cached, not used: no recruiting data) | `RecruitingApi.get_team_recruiting_rankings(year)` | 17 (2010–2026; 4-year averages need 2010–13) |
| Talent composite (cached, not used: built from recruiting ratings) | `TeamsApi.get_talent(year)` | 13 (2014–2026) |
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
2. **Preseason strength:** returning production (share of last season's EPA back,
   clipped to [0, 1]). It also gets a version multiplied by `1 / (1 + games played)`, so
   it matters in September and fades as games accumulate. Last season's final rating
   is already inside the ratings group (it is their prior).
   **No recruiting data (decision, Oct 27, 2026):** recruiting class rankings and the
   talent composite (built from players' recruiting ratings) are not used as features.
   Their loaders stay in `data.py`, and the cached files are simply unused.
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

## 5b results (Oct 27, 2026)

`scripts/evaluate_features.py`, walk-forward, 2016-2024 excl. 2020, 7,079 games with a
closing line. The stack is a ridge regression of margin on standardized features,
trained on earlier seasons only; each group is kept only if log loss improves.

| Model | Log loss | Brier | MAE | Log loss, weeks 1-4 |
|---|---|---|---|---|
| Baseline (Phase 4 ridge + normal) | 0.5005 | 0.1676 | 13.318 | 0.4141 |
| Stack: ratings only | 0.5010 | 0.1677 | 13.278 | 0.4150 |
| + efficiency (EPA, success rate) | 0.4994 | 0.1671 | 13.261 | 0.4133 |
| + preseason (returning production) | 0.4958 | 0.1656 | 13.180 | 0.4110 |
| + situational (rest, travel) | **0.4957** | **0.1655** | **13.177** | **0.4107** |
| Market (closing line) | 0.4758 | 0.1586 | 12.302 | 0.3760 |

- Log loss 0.5005 -> 0.4957 closes about a fifth of the gap to the market (0.0247 -> 0.0199).
- Returning production helped most. Situational features barely moved log loss
  (0.0001, within noise), but were kept by the rule; rest and travel are significant
  in the coefficient table (t = 1.9 and 3.0).
- Efficiency ratings: EPA alpha 10, revert 0.1; success rate alpha 3, revert 0.3
  (tuned by pre-game prediction error). 2024 ratings pass a sanity check (top
  offenses Ohio State, Oregon, Alabama; top defenses Notre Dame, Texas, Penn State).
- Calibration: `reports/figures/stack_calibration.png`, close to the diagonal.
- **2025 test season (scored once, Oct 27, 2026): the gain did not hold up.**

  | 2025, 934 games with a line | Log loss | Brier | MAE | Log loss, weeks 1-4 |
  |---|---|---|---|---|
  | Baseline (Phase 4) | **0.4758** | **0.1589** | 12.365 | **0.3694** |
  | Stack (all four groups) | 0.4781 | 0.1595 | **12.340** | 0.3760 |
  | Market (closing line) | 0.4668 | 0.1565 | 11.897 | 0.3465 |

  Paired per-game difference, stack minus baseline (SE = sd / sqrt(n)): tuning seasons
  -0.0048 (SE 0.0014, t = -3.5); 2025 +0.0023 (SE 0.0034, t = +0.7). One season can't
  separate a small gain from zero, but the tuning-season gain sits about 2 SE below the
  2025 result, so some of it was probably optimism from tuning and choosing groups on
  the same seasons. Weeks 1-4 were worse in 2025 (+0.0065, SE 0.0062), the part that
  returning production drives.
  **Decision:** the Phase 4 model stays the live model. Don't re-tune on 2025 (that would
  turn the test season into a tuning season). The 2026 games are the next honest test.
- **Shadow mode (from 2026 week 10):** `predict_week.py` records the stack's spread and
  win probability next to the live ridge model in every prediction file;
  `score_week.py` prints a "shadow check" (ridge vs. stack vs. closing line on the same
  games) for the week and the season so far. Live features are built by the same
  functions as the backtest; with 2025 week 8 treated as unplayed, they match the
  backtest features exactly. 2026 advanced stats are fetched one finished week at a
  time (1 API call per week).

## 5c results (Oct 28, 2026)

`scripts/evaluate_phase5.py`, same games and splits as 5b (2016-2024 excl. 2020, 7,079
games with a closing line). Every model uses the same 10 features (all four 5b groups).
**Change from the plan above:** feature groups were not re-selected one at a time for
each new model. 2025 suggested the 5b selection was partly optimism, so each model only
tunes its own hyperparameters.

| Model | Log loss | Brier | MAE | Log loss, weeks 1-4 | vs. baseline (SE) |
|---|---|---|---|---|---|
| Baseline (Phase 4 ridge + normal) | 0.5005 | 0.1676 | 13.318 | 0.4141 | — |
| Elo | 0.5182 | 0.1744 | — | 0.4373 | +0.0178 (0.0027) |
| Stack (ridge, alpha 100) | **0.4957** | 0.1655 | **13.177** | **0.4107** | -0.0048 (0.0014) |
| Penalized logistic (C 0.01) | **0.4957** | **0.1654** | — | 0.4109 | -0.0047 (0.0015) |
| LightGBM (4 leaves, rate 0.01, 400 trees) | 0.4977 | 0.1662 | 13.234 | 0.4140 | -0.0028 (0.0015) |
| Market (closing line) | 0.4758 | 0.1586 | 12.302 | 0.3760 | |

- **Logistic ties the stack.** Modeling the win directly (no normal assumption, no
  sigma) gives the same answer as modeling the margin and converting it, which means
  Phi(spread / sigma) was not losing anything. Unpenalized logit coefficients tell the
  same story as the stack's OLS table: returning production (z = 5.6), success rate
  (5.5), Elo (3.9), and travel (2.6) are clearly nonzero; the ridge and Huber spreads
  overlap so much that neither is individually significant.
- **LightGBM is worse than the linear models**, by about 0.002, at every setting tried.
  The best settings are the simplest: a first grid (4-16 leaves, rate 0.02-0.05,
  200-500 trees) was best at its simplest corner, and a second, simpler grid (2-8
  leaves, rate 0.01-0.02, 100-400 trees) plateaued at 0.4977. Stumps (one split per
  tree, an additive model with no interactions) reach the same 0.4977. More leaves or
  more trees only add overfitting. With about 6,000 training games and features that
  are already model outputs, there are no interactions or nonlinearities for the trees
  to find that the linear model misses. Gain importance: Huber spread 47%, ridge spread
  29%, success rate 12%, Elo 10%; the rest about 2% combined.
- **LightGBM's sigma** comes from the last training season held out (fit on the seasons
  before it): 16.2-17.6 points by season, close to the stack's in-sample 16.6. 2016
  falls back to fitting on the 2014 burn-in season (17.4, slightly conservative).
- Calibration: `reports/figures/logit_calibration.png` and `lgbm_calibration.png`, both
  close to the diagonal.
- **2025 test season (scored once, Oct 28, 2026): no model beats the baseline.**

  | 2025, 934 games with a line | Log loss | Brier | MAE | Log loss, weeks 1-4 | vs. baseline (SE) |
  |---|---|---|---|---|---|
  | Baseline (Phase 4) | **0.4758** | **0.1589** | 12.365 | **0.3694** | — |
  | Elo | 0.4998 | 0.1677 | — | 0.4145 | +0.0240 (0.0074) |
  | Stack | 0.4781 | 0.1595 | **12.340** | 0.3760 | +0.0023 (0.0034) |
  | Penalized logistic | 0.4792 | 0.1598 | — | 0.3762 | +0.0034 (0.0042) |
  | LightGBM | 0.4791 | 0.1601 | 12.424 | 0.3772 | +0.0033 (0.0034) |
  | Market (closing line) | 0.4668 | 0.1565 | 11.897 | 0.3465 | |

  All three Phase 5 models are within one SE of each other and of the baseline. The
  differences are not meaningful.

**Decision:** the Phase 4 model stays the live model and the stack stays the shadow
model. Logistic and LightGBM are kept in `models.py` for comparison but are not added to
the weekly predictions: on the tuning seasons neither beats the stack, so they add
nothing to test. The remaining gap to the market (0.020 log loss on the tuning seasons)
looks like a gap in information (injuries, quarterback changes, depth charts), not in
model flexibility.

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
