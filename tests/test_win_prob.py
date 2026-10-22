import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from football_model import config, win_prob


def make_games(rows: list[tuple]) -> pd.DataFrame:
    """Build a games table from (season, week, season_type, margin, spread) tuples."""
    games = pd.DataFrame(rows, columns=["season", "week", "season_type", "margin", "spread"])
    games["home_win"] = (games["margin"] > 0).astype(int)
    return games


# Two seasons. 2023 residuals (margin - spread): week 1 -> +10, -10; week 9 -> +2, -2.
GAMES = make_games(
    [
        (2023, 1, "regular", 13, 3.0),
        (2023, 1, "regular", -7, 3.0),
        (2023, 9, "regular", 7, 5.0),
        (2023, 9, "regular", 3, 5.0),
        (2024, 1, "regular", 10, 0.0),
        (2024, 9, "regular", -3, 7.0),
    ]
)


def test_residual_sd_hand_calculation():
    # Residuals 10, -10, 2, -2: mean 0, sample variance (100+100+4+4)/3 = 69.33.
    assert win_prob.residual_sd(GAMES, "spread", [2023]) == pytest.approx(np.sqrt(208 / 3))


def test_walk_forward_sigma_uses_earlier_seasons_only():
    sigma = win_prob.walk_forward_sigma(GAMES, "spread")
    assert sigma[GAMES["season"] == 2023].isna().all()
    assert sigma[GAMES["season"] == 2024].tolist() == pytest.approx([np.sqrt(208 / 3)] * 2)


def test_by_week_sigma_uses_matching_bucket():
    sigma = win_prob.walk_forward_sigma(GAMES, "spread", by_week=True)
    # Week-1 residuals 10, -10 -> SD sqrt(200); week-9 residuals 2, -2 -> SD sqrt(8).
    assert sigma.iloc[4] == pytest.approx(np.sqrt(200))
    assert sigma.iloc[5] == pytest.approx(np.sqrt(8))


def test_normal_prob_sign_and_symmetry():
    games = make_games(
        [
            (2023, 1, "regular", 10, 0.0),
            (2023, 1, "regular", -10, 0.0),
            (2024, 1, "regular", 1, 0.0),
            (2024, 1, "regular", 1, 7.0),
            (2024, 1, "regular", 1, -7.0),
        ]
    )
    p = win_prob.normal_prob(games, "spread")
    assert p.iloc[2] == pytest.approx(0.5)
    assert p.iloc[3] > 0.5
    assert p.iloc[3] + p.iloc[4] == pytest.approx(1.0)
    assert p.iloc[3] == pytest.approx(norm.cdf(7.0 / np.sqrt(200)))


def test_no_leakage_from_current_or_later_seasons():
    base_sigma = win_prob.walk_forward_sigma(GAMES, "spread")
    base_logit = win_prob.logistic_prob(simulated_games(), "spread")

    # Changing 2024 results must not change 2024's sigma (it comes from 2023).
    changed = GAMES.copy()
    changed.loc[changed["season"] == 2024, "margin"] = [50, -50]
    assert win_prob.walk_forward_sigma(changed, "spread").equals(base_sigma)

    # Same for the logistic fit: flip the last season's outcomes.
    sim = simulated_games()
    last = sim["season"] == sim["season"].max()
    sim.loc[last, "home_win"] = 1 - sim.loc[last, "home_win"]
    after = win_prob.logistic_prob(sim, "spread")
    assert after.equals(base_logit)


def simulated_games(a: float = 0.0, b: float = 0.1, n: int = 4000) -> pd.DataFrame:
    """Simulate home_win from a known logistic model, two seasons, fixed seed."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    spread = rng.normal(0, 15, n)
    p = 1 / (1 + np.exp(-(a + b * spread)))
    home_win = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(
        {
            "season": np.repeat([2023, 2024], n // 2),
            "spread": spread,
            "home_win": home_win,
            "margin": np.where(home_win == 1, 1, -1),
        }
    )


def test_logistic_recovers_known_coefficients():
    sim = simulated_games(a=0.2, b=0.1)
    result = win_prob.fit_logistic(sim, "spread", [2023, 2024])
    assert result.params["const"] == pytest.approx(0.2, abs=0.1)
    assert result.params["spread"] == pytest.approx(0.1, abs=0.01)


def test_probabilities_strictly_between_zero_and_one():
    sim = simulated_games()
    sim["week"] = 5
    sim["season_type"] = "regular"
    out = win_prob.add_win_probs(sim, "spread", "m")
    scored = out[out["season"] == 2024]
    for col in ["m_prob_normal", "m_prob_normal_by_week", "m_prob_logistic"]:
        assert scored[col].between(0, 1, inclusive="neither").all()
