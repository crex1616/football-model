"""Monte Carlo simulation of the rest of a season: wins, bowls, conference titles, playoff.

The statistical idea, in three steps:

1. Ratings with honest uncertainty. The ridge ratings (spread.py) are the posterior mean
   of a Bayesian regression; their posterior covariance is

       Cov(b) = sigma_e^2 (X'X + alpha I)^-1

   (the OLS covariance sigma^2 (X'X)^-1 with the penalty added). Each simulated season
   draws one full set of team ratings from N(b_hat, Cov(b)). A team that is "really"
   better in one draw is better in all of its remaining games, so its results are
   correlated the way real seasons are. This is how ratings "move" within a simulation.

2. Game noise without double counting. sigma_oos, the out-of-sample residual SD used for
   win probabilities, already contains rating error. So the per-game noise is the part
   left over:  sigma_oos^2 = sigma_e^2 * (1 + m),  where m is the average of
   c'(X'X + alpha I)^-1 c over the remaining games (c = a game's row of the design
   matrix). Solving gives sigma_e = sigma_oos / sqrt(1 + m).

3. Rules applied to every simulated season: conference standings (simplified
   tiebreakers), conference championship games, and an approximate playoff selection.

Simplifications (documented so results are read honestly):
- Tiebreakers: conference win %, then head-to-head for an exact two-team tie, then the
  higher simulated rating (a stand-in for the official multi-step procedures).
- Playoff (2026 format, 12 teams): ACC, Big 12, Big Ten, SEC champions plus the best-
  ranked champion of the other six conferences, plus 7 at-large. "Ranked" is a proxy
  for the committee, two rules kept side by side:
  - résumé (headline, p_playoff_resume): lowest strength of record (rankings.py), the
    better match to the committee's final rankings (Spearman 0.904 vs. 0.790 on
    2015-2024 excl. 2020; 2025 test 0.934 vs. 0.772; scripts/evaluate_rankings.py);
  - record (p_playoff): fewest losses, then higher simulated rating.
- Bowl eligibility: 6 wins, with at most one win over an FCS team counting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from football_model import config, predict, rankings, spread

FCS_TEAM = spread.FCS_TEAM
HOME_FIELD = spread.HOME_FIELD
INDEPENDENTS = "FBS Independents"
POWER_CONFERENCES = ("ACC", "Big 12", "Big Ten", "SEC")
FIRST_SEASON_WITH_NOTES = 2022  # CFBD labels title games in `notes` from here on
PLAYOFF_TEAMS = 12
BOWL_WINS = 6
NINE_WINS = 9
RIDGE = spread.SpreadParams()  # the live model: plain ridge
# A conference this small has no title game or champion here (the 2024 Pac-12 had two).
MIN_CONFERENCE_SIZE = 4


# --- Schedule helpers ---------------------------------------------------------------


def is_title_game(games: pd.DataFrame) -> pd.Series:
    """True for conference championship games.

    CFBD labels them in `notes` from 2022 on ("SEC Championship"), so from 2022 only the
    label counts. Earlier seasons have no notes, so a title game is inferred: a December
    conference game between two teams of the same conference that is the only
    conference game of that conference in its week. (The inference is not used from
    2022 on: Army-Navy, an American conference game since 2024, is played alone in
    December and would look like a title game.)
    """
    same_conf = games["conference_game"].fillna(False).astype(bool) & (
        games["home_conference"] == games["away_conference"]
    )
    noted = games["notes"].fillna("").str.contains("Championship") & same_conf
    december = games["start_date"].dt.month == 12
    per_week = games[same_conf].groupby(["season", "week", "home_conference"])["game_id"]
    only_one = games.set_index(["season", "week", "home_conference"]).index.map(
        per_week.count().to_dict()
    )
    inferred = same_conf & december & (pd.Series(only_one, index=games.index) == 1)
    return noted | (inferred & (games["season"] < FIRST_SEASON_WITH_NOTES))


def team_conferences(games: pd.DataFrame) -> dict[str, str]:
    """Each FBS team's conference this season, from its games (latest game wins)."""
    home = games[["start_date", "home_team", "home_conference"]].set_axis(
        ["start_date", "team", "conference"], axis=1
    )
    away = games[["start_date", "away_team", "away_conference"]].set_axis(
        ["start_date", "team", "conference"], axis=1
    )
    both = pd.concat([home, away]).sort_values("start_date")
    both = both[both["team"] != FCS_TEAM].drop_duplicates("team", keep="last")
    return dict(zip(both["team"], both["conference"]))


def game_contrasts(games: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Each game's row of the ridge design matrix, columns in `index` order.

    home_field column = 1 (0 at a neutral site), home team +1, away team -1.
    """
    teams = list(index.drop(HOME_FIELD))
    X, _ = spread.design_matrix(games.assign(margin=np.nan), teams)
    return X[list(index)].to_numpy()


# --- Ratings and noise --------------------------------------------------------------


def ridge_posterior(
    train: pd.DataFrame, prior: pd.Series, alpha: float
) -> tuple[pd.Series, np.ndarray]:
    """Posterior mean and unscaled covariance of the ridge ratings.

    Args:
        train: This season's completed games.
        prior: Prior mean for each coefficient (spread.season_prior).
        alpha: Ridge penalty.

    Returns:
        (mean, U): mean = m + (X'X + alpha I)^-1 X'(y - X m), identical to
        spread.fit_ratings with plain ridge; U = (X'X + alpha I)^-1, so that
        Cov(b) = sigma_e^2 * U.
    """
    teams = list(prior.index.drop(HOME_FIELD))
    if train.empty:
        return prior.copy(), np.eye(len(prior)) / alpha
    X, y = spread.design_matrix(train, teams)
    x = X.to_numpy()
    precision = x.T @ x + alpha * np.eye(x.shape[1])
    mean = spread.ridge_solve(x, y.to_numpy(), np.ones(len(y)), prior.to_numpy(), alpha)
    return pd.Series(mean, index=prior.index), np.linalg.inv(precision)


def split_noise(sigma_oos: float, unscaled_cov: np.ndarray, contrasts: np.ndarray) -> float:
    """Per-game noise SD once rating uncertainty is accounted for separately.

    sigma_oos^2 = sigma_e^2 (1 + m), m = mean over games of c' U c.
    """
    if len(contrasts) == 0:
        return sigma_oos
    m = float(np.mean(np.einsum("gi,ij,gj->g", contrasts, unscaled_cov, contrasts)))
    return sigma_oos / np.sqrt(1.0 + m)


def season_state(
    history: pd.DataFrame,
    teams: list[str],
    season: int,
    params: spread.SpreadParams = RIDGE,
    fits: dict[int, pd.Series] | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """Posterior ratings for `season` given every completed game so far.

    The prior comes from last season's final fit, as in spread.run_spread.

    Args:
        history: Completed games, earlier seasons plus this season so far.
        teams: Every team appearing in this season's schedule (played or not).
        season: The season being simulated.
        params: Plain ridge settings (no Huber, cap, or recency weights).
        fits: spread.season_fits output, if already computed (saves time in a backtest;
            only last season's fit is used).

    Returns:
        (mean, U) from ridge_posterior.
    """
    if params.huber_c or params.margin_cap or params.half_life_weeks:
        raise ValueError("season_state needs the plain ridge model")
    if fits is None:
        fits = spread.season_fits(history[history["season"] < season], params)
    prior = spread.season_prior(fits.get(season - 1), teams, params.revert)
    this_season = history[history["season"] == season]
    return ridge_posterior(this_season, prior, params.alpha)


# --- Standings ----------------------------------------------------------------------


def order_conference(pct: np.ndarray, rating: np.ndarray, beat: np.ndarray) -> np.ndarray:
    """Order one conference's teams in one simulation, best first.

    Sort by conference win %, then simulated rating. Where exactly two teams share a
    win % at the top-two boundaries, the head-to-head winner goes first.

    Args:
        pct: Conference win % per team.
        rating: Simulated rating per team (last-resort tiebreaker).
        beat: beat[i, j] = 1 if i beat j head-to-head, -1 if j beat i, 0 if they didn't play.

    Returns:
        Team positions (into pct), best first.
    """
    order = np.lexsort((-rating, -pct))
    for k in (1, 0):
        if k + 1 >= len(order):
            continue
        a, b = order[k], order[k + 1]
        exact_two = pct[a] == pct[b] and int(np.sum(pct == pct[a])) == 2
        if exact_two and beat[b, a] == 1:
            order[k], order[k + 1] = b, a
    return order


def _incidence(games: pd.DataFrame, teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Home and away incidence matrices (teams x games): 1 where the team plays."""
    position = {t: i for i, t in enumerate(teams)}
    home = np.zeros((len(teams), len(games)))
    away = np.zeros((len(teams), len(games)))
    for g, (h, a) in enumerate(zip(games["home_team"], games["away_team"])):
        if h in position:
            home[position[h], g] = 1.0
        if a in position:
            away[position[a], g] = 1.0
    return home, away


def simulate_season(
    completed: pd.DataFrame,
    remaining: pd.DataFrame,
    mean: pd.Series,
    unscaled_cov: np.ndarray,
    sigma_oos: float,
    conferences: dict[str, str],
    divisions: dict[str, str] | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate the rest of a season n_sims times and summarize each FBS team's odds.

    Args:
        completed: This season's completed games (clean_games output).
        remaining: This season's unplayed regular-season games, including any scheduled
            title games (clean_schedule output).
        mean, unscaled_cov: ridge_posterior output.
        sigma_oos: Out-of-sample residual SD of the spread model (win_prob.current_sigma).
        conferences: FBS team -> conference (team_conferences).
        divisions: Team -> division, for conferences that still have them (Sun Belt).
        n_sims: Number of simulated seasons.
        seed: Random seed (config.RANDOM_SEED in scripts).

    Returns:
        One row per FBS team: team, conference, wins, losses (so far), exp_regular_wins,
        exp_wins (including a title game), p_9_wins (9+ wins, title game included, bowls
        not simulated), p_bowl, p_title_game, p_conf_champ, p_playoff (record rule),
        p_playoff_resume (résumé rule, the headline).
    """
    divisions = divisions or {}
    rng = np.random.default_rng(seed)
    teams = sorted(conferences)
    t_pos = {t: i for i, t in enumerate(teams)}
    index = mean.index

    title_done = completed[is_title_game(completed)]
    title_left = remaining[is_title_game(remaining)]
    # A scheduled title game counts only once both teams are known FBS members.
    title_left = title_left[
        title_left["home_team"].isin(teams) & title_left["away_team"].isin(teams)
    ]
    regular_left = remaining[~is_title_game(remaining)]
    regular_left = regular_left[
        ~((regular_left["home_team"] == FCS_TEAM) & (regular_left["away_team"] == FCS_TEAM))
    ]
    regular_done = completed[~is_title_game(completed)]

    # Ratings: one posterior draw per simulation; game noise net of rating uncertainty.
    contrasts = game_contrasts(regular_left, index)
    sigma_e = split_noise(sigma_oos, unscaled_cov, contrasts)
    chol = np.linalg.cholesky(sigma_e**2 * unscaled_cov + 1e-9 * np.eye(len(index)))
    draws = mean.to_numpy()[:, None] + chol @ rng.standard_normal((len(index), n_sims))
    team_rating = draws[[index.get_loc(t) for t in teams], :]

    # Remaining regular-season games: home win (games x sims).
    expected = contrasts @ draws
    home_win = (expected + rng.normal(0.0, sigma_e, expected.shape)) > 0

    # Records: completed (constant) + simulated.
    home_inc, away_inc = _incidence(regular_left, teams)
    wins = (home_inc @ home_win + away_inc @ ~home_win).astype(float)
    played_left = (home_inc + away_inc).sum(axis=1)[:, None]
    losses = played_left - wins
    vs_fcs = (regular_left["home_team"] == FCS_TEAM) | (regular_left["away_team"] == FCS_TEAM)
    fcs_wins = (home_inc[:, vs_fcs.to_numpy()] @ home_win[vs_fcs.to_numpy()]) + (
        away_inc[:, vs_fcs.to_numpy()] @ ~home_win[vs_fcs.to_numpy()]
    )

    done_wins, done_losses, done_fcs_wins = _completed_records(regular_done, teams)
    wins += done_wins[:, None]
    losses += done_losses[:, None]
    fcs_wins = fcs_wins + done_fcs_wins[:, None]
    regular_wins = wins.copy()

    # Conference standings.
    conf_game = (
        regular_left["conference_game"].fillna(False).astype(bool)
        & (regular_left["home_conference"] == regular_left["away_conference"])
    ).to_numpy()
    c_home, c_away = home_inc[:, conf_game], away_inc[:, conf_game]
    c_win = c_home @ home_win[conf_game] + c_away @ ~home_win[conf_game]
    c_played = (c_home + c_away).sum(axis=1)[:, None]
    done_cw, done_cl = _completed_conference_records(regular_done, teams)
    conf_w = c_win + done_cw[:, None]
    conf_l = c_played - c_win + done_cl[:, None]
    conf_pct = np.divide(
        conf_w, conf_w + conf_l, out=np.zeros_like(conf_w), where=(conf_w + conf_l) > 0
    )

    in_title_game = np.zeros((len(teams), n_sims), dtype=bool)
    champion = np.zeros((len(teams), n_sims), dtype=bool)
    title_opponent = np.full((len(teams), n_sims), -1)  # for the résumé rule's SOR

    for conference in sorted(set(conferences.values()) - {INDEPENDENTS}):
        members = [t for t in teams if conferences[t] == conference]
        if len(members) < MIN_CONFERENCE_SIZE:
            continue
        rows = np.array([t_pos[t] for t in members])
        done = title_done[title_done["home_conference"] == conference]
        if len(done):  # the real title game has been played
            game = done.iloc[-1]
            a, b = t_pos[game["home_team"]], t_pos[game["away_team"]]
            in_title_game[[a, b], :] = True
            title_opponent[a, :], title_opponent[b, :] = b, a
            champion[a if game["margin"] > 0 else b, :] = True
            continue

        scheduled = title_left[title_left["home_conference"] == conference]
        if len(scheduled):  # matchup known, result not
            game = scheduled.iloc[-1]
            a = np.full(n_sims, t_pos[game["home_team"]])
            b = np.full(n_sims, t_pos[game["away_team"]])
        else:
            beat = _head_to_head(completed, regular_left, home_win, members)
            a, b = _pick_title_teams(members, rows, conf_pct, team_rating, beat, divisions, n_sims)

        sims = np.arange(n_sims)
        in_title_game[a, sims] = True
        in_title_game[b, sims] = True
        title_opponent[a, sims], title_opponent[b, sims] = b, a
        margin = team_rating[a, sims] - team_rating[b, sims] + rng.normal(0.0, sigma_e, n_sims)
        winner = np.where(margin > 0, a, b)
        loser = np.where(margin > 0, b, a)
        champion[winner, sims] = True
        wins[winner, sims] += 1
        losses[loser, sims] += 1

    playoff = _playoff_field(teams, conferences, champion, losses, team_rating)
    sor = _simulated_sor(
        pd.concat([regular_done, regular_left], ignore_index=True),
        teams,
        mean,
        sigma_oos,
        wins,
        title_opponent,
    )
    playoff_resume = _playoff_field(teams, conferences, champion, losses, team_rating, score=sor)
    counted = wins - np.maximum(fcs_wins - 1, 0)

    current_w, current_l, _ = _completed_records(completed, teams)
    return pd.DataFrame(
        {
            "team": teams,
            "conference": [conferences[t] for t in teams],
            "wins": current_w.astype(int),
            "losses": current_l.astype(int),
            "exp_regular_wins": regular_wins.mean(axis=1),
            "exp_wins": wins.mean(axis=1),
            "p_9_wins": (wins >= NINE_WINS).mean(axis=1),
            "p_bowl": (counted >= BOWL_WINS).mean(axis=1),
            "p_title_game": in_title_game.mean(axis=1),
            "p_conf_champ": champion.mean(axis=1),
            "p_playoff": playoff.mean(axis=1),
            "p_playoff_resume": playoff_resume.mean(axis=1),
        }
    )


def _simulated_sor(
    regular_schedule: pd.DataFrame,
    teams: list[str],
    mean: pd.Series,
    sigma: float,
    wins: np.ndarray,
    title_opponent: np.ndarray,
) -> np.ndarray:
    """Strength of record (teams x sims) for each simulated final record.

    The benchmark team (average of the power top 25, posterior-mean ratings) plays each
    team's regular-season schedule; that win distribution is fixed, so SOR is a lookup
    at the simulated win total. A title game is one more Bernoulli trial with win
    probability p against that simulation's opponent (neutral site):
    P(W >= w) = p * P_reg(W >= w - 1) + (1 - p) * P_reg(W >= w).

    Simplification: the benchmark uses today's ratings, not ratings refit after the
    simulated results.
    """
    benchmark = rankings.benchmark_rating(mean, teams)
    max_wins = int(wins.max()) + 1
    tails = rankings.schedule_tails(regular_schedule, teams, mean, sigma, benchmark, max_wins)
    w = wins.astype(int)
    rows = np.arange(len(teams))[:, None]
    sor = tails[rows, w]
    has_title = title_opponent >= 0
    team_mean = mean[teams].to_numpy()
    opp_rating = team_mean[np.where(has_title, title_opponent, 0)]
    p = norm.cdf((benchmark - opp_rating) / sigma)
    with_title = p * tails[rows, np.maximum(w - 1, 0)] + (1 - p) * sor
    return np.where(has_title, with_title, sor)


def _completed_records(
    games: pd.DataFrame, teams: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wins, losses, and wins over FCS teams in completed games."""
    wins, losses, fcs_wins = (np.zeros(len(teams)) for _ in range(3))
    position = {t: i for i, t in enumerate(teams)}
    for h, a, m in zip(games["home_team"], games["away_team"], games["margin"]):
        for team, opp, won in ((h, a, m > 0), (a, h, m < 0)):
            if team in position:
                i = position[team]
                wins[i] += won
                losses[i] += not won
                fcs_wins[i] += won and opp == FCS_TEAM
    return wins, losses, fcs_wins


def _completed_conference_records(
    games: pd.DataFrame, teams: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Conference wins and losses in completed (non-title) conference games."""
    same = games["conference_game"].fillna(False).astype(bool) & (
        games["home_conference"] == games["away_conference"]
    )
    wins, losses, _ = _completed_records(games[same], teams)
    return wins, losses


def _head_to_head(
    completed: pd.DataFrame, remaining: pd.DataFrame, home_win: np.ndarray, members: list[str]
) -> np.ndarray:
    """beat[i, j, s] = 1 if member i beat member j in simulation s, -1 if j beat i."""
    position = {t: i for i, t in enumerate(members)}
    n_sims = home_win.shape[1]
    beat = np.zeros((len(members), len(members), n_sims), dtype=np.int8)
    for h, a, m in zip(completed["home_team"], completed["away_team"], completed["margin"]):
        if h in position and a in position:
            sign = 1 if m > 0 else -1
            beat[position[h], position[a], :] = sign
            beat[position[a], position[h], :] = -sign
    for g, (h, a) in enumerate(zip(remaining["home_team"], remaining["away_team"])):
        if h in position and a in position:
            sign = np.where(home_win[g], 1, -1).astype(np.int8)
            beat[position[h], position[a], :] = sign
            beat[position[a], position[h], :] = -sign
    return beat


def _pick_title_teams(
    members: list[str],
    rows: np.ndarray,
    conf_pct: np.ndarray,
    team_rating: np.ndarray,
    beat: np.ndarray,
    divisions: dict[str, str],
    n_sims: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The two title-game teams in each simulation (team rows into the full team list).

    Top two in the standings, or each division's winner if the conference has
    divisions.
    """
    member_divs = [divisions.get(t) for t in members]
    split = len({d for d in member_divs if d}) == 2
    a = np.empty(n_sims, dtype=int)
    b = np.empty(n_sims, dtype=int)
    pct, rating = conf_pct[rows], team_rating[rows]
    for s in range(n_sims):
        if split:
            picks = []
            for div in sorted({d for d in member_divs if d}):
                idx = np.array([i for i, d in enumerate(member_divs) if d == div])
                sub = order_conference(
                    pct[idx, s], rating[idx, s], beat[np.ix_(idx, idx, [s])][:, :, 0]
                )
                picks.append(idx[sub[0]])
            a[s], b[s] = rows[picks[0]], rows[picks[1]]
        else:
            order = order_conference(pct[:, s], rating[:, s], beat[:, :, s])
            a[s], b[s] = rows[order[0]], rows[order[1]]
    return a, b


def _playoff_field(
    teams: list[str],
    conferences: dict[str, str],
    champion: np.ndarray,
    losses: np.ndarray,
    rating: np.ndarray,
    score: np.ndarray | None = None,
) -> np.ndarray:
    """Playoff indicator (teams x sims) under the 2026 rule.

    Rank: lowest `score` (the résumé rule passes SOR; None means the record rule), then
    fewest losses, then higher simulated rating. Field: the four power-conference
    champions, the best-ranked other champion, then the best-ranked others to 12.
    """
    n_teams, n_sims = champion.shape
    score = losses if score is None else score
    power = np.array([conferences[t] in POWER_CONFERENCES for t in teams])
    other_conf = np.array(
        [conferences[t] not in POWER_CONFERENCES and conferences[t] != INDEPENDENTS for t in teams]
    )
    field = np.zeros((n_teams, n_sims), dtype=bool)
    for s in range(n_sims):
        order = np.lexsort((-rating[:, s], losses[:, s], score[:, s]))
        chosen = set(np.flatnonzero(champion[:, s] & power))
        other_champs = [i for i in order if champion[i, s] and other_conf[i]]
        if other_champs:
            chosen.add(other_champs[0])
        for i in order:
            if len(chosen) >= PLAYOFF_TEAMS:
                break
            chosen.add(i)
        field[list(chosen), s] = True
    return field


# --- Output -------------------------------------------------------------------------

ODDS_PERCENT_COLUMNS = [
    "Win 9+",
    "Bowl eligible (6+)",
    "Title game",
    "Win conference",
    "Playoff",
    "Playoff (record rule)",
]
ODDS_NUMBER_COLUMNS = ["Expected wins"]


def odds_path(season: int, after_week: int, suffix: str = ".csv") -> Path:
    """predictions/{season}/season_odds_after_week_{WW}{suffix}: a dated snapshot."""
    name = f"season_odds_after_week_{after_week:02d}{suffix}"
    return config.PREDICTIONS_DIR / str(season) / name


def same_odds(saved: pd.DataFrame, new: pd.DataFrame) -> bool:
    """True if a new simulation reproduces a saved snapshot on every column they share.

    Used to rebuild a snapshot's .xlsx (for example with a new column) without editing
    the saved CSV: same games played and same seed give the same numbers.
    """
    shared = [c for c in saved.columns if c in new.columns]
    if len(saved) != len(new) or not shared:
        return False
    for column in shared:
        a = saved[column].reset_index(drop=True)
        b = new[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a, b):
                return False
        elif not a.astype(str).equals(b.astype(str)):
            return False
    return True


def odds_table(odds: pd.DataFrame) -> pd.DataFrame:
    """Readable odds, best playoff odds first: team and record, then the probabilities.

    Expected wins include a conference title game (bowls aren't simulated). "Playoff"
    is the résumé rule, the better committee proxy; "Playoff (record rule)" is the
    original one. Snapshots saved before those columns existed (2026 after week 8) show
    9+ blank and the record rule as "Playoff".
    """
    odds = odds.copy()
    if "p_9_wins" not in odds.columns:
        odds["p_9_wins"] = float("nan")
    if "p_playoff_resume" in odds.columns:
        headline, other = odds["p_playoff_resume"], odds["p_playoff"]
    else:
        headline, other = odds["p_playoff"], pd.Series(float("nan"), index=odds.index)
    odds = odds.assign(headline=headline, other=other)
    ordered = odds.sort_values(["headline", "p_conf_champ", "exp_wins"], ascending=False)
    return pd.DataFrame(
        {
            "Team": ordered["team"],
            "Conference": ordered["conference"],
            "Record": [f"{w}-{l}" for w, l in zip(ordered["wins"], ordered["losses"])],
            "Expected wins": ordered["exp_wins"],
            "Win 9+": ordered["p_9_wins"],
            "Bowl eligible (6+)": ordered["p_bowl"],
            "Title game": ordered["p_title_game"],
            "Win conference": ordered["p_conf_champ"],
            "Playoff": ordered["headline"],
            "Playoff (record rule)": ordered["other"],
        }
    ).reset_index(drop=True)


def odds_tabs(odds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split odds_table into tabs: "Playoff odds" for every team, then one per conference.

    The playoff tab is ranked by playoff odds. Each conference tab is ranked by odds to
    win the conference and leaves out the (repeated) conference column; the
    independents' tab also leaves out the title-game columns, since they have none.
    """
    table = odds_table(odds)
    tabs = {"Playoff odds": table.assign(Rank=range(1, len(table) + 1))}
    tabs["Playoff odds"] = tabs["Playoff odds"][["Rank", *table.columns]]
    # Conferences alphabetically, independents last.
    for conference in sorted(table["Conference"].unique(), key=lambda c: (c == INDEPENDENTS, c)):
        rows = table[table["Conference"] == conference]
        rows = rows.sort_values(["Win conference", "Expected wins"], ascending=False)
        rows = rows.drop(columns="Conference")
        if conference == INDEPENDENTS:
            rows = rows.drop(columns=["Title game", "Win conference"])
        # Excel sheet names: at most 31 characters, no / \ ? * [ ] :
        name = "".join(ch for ch in conference if ch not in "/\\?*[]:")[:31]
        tabs[name] = rows.reset_index(drop=True)
    return tabs


def write_odds_excel(odds: pd.DataFrame, path: Path) -> None:
    """Excel copy of the odds: a "Playoff odds" tab, then one tab per conference.

    Styled like the weekly predictions workbook (bold headers, fitted widths, whole
    percents), with the expected-wins column highlighted. The CSV stays the record.
    """
    predict.write_tabs(
        odds_tabs(odds),
        path,
        ODDS_PERCENT_COLUMNS,
        ODDS_NUMBER_COLUMNS,
        freeze="B2",
        highlight_columns=["Expected wins"],
    )
