import math

import numpy as np
import pandas as pd
import pytest

from football_model import evaluate


def test_log_loss_hand_calculation():
    y = [1, 0]
    p = [0.8, 0.4]
    # -(ln 0.8 + ln 0.6) / 2
    expected = -(math.log(0.8) + math.log(0.6)) / 2
    assert evaluate.log_loss(y, p) == pytest.approx(expected)


def test_coin_flip_log_loss_and_brier():
    y = [1, 0, 1, 0]
    p = [0.5] * 4
    assert evaluate.log_loss(y, p) == pytest.approx(math.log(2))
    assert evaluate.brier_score(y, p) == pytest.approx(0.25)


def test_market_win_prob_sign_convention():
    # Positive spread = home favored, so home win probability is above 0.5.
    probs = evaluate.market_win_prob(np.array([7.0, 0.0, -7.0]), sigma=15.0)
    assert probs[0] > 0.5
    assert probs[1] == pytest.approx(0.5)
    assert probs[2] == pytest.approx(1 - probs[0])


def test_score_seasons_skips_games_without_a_line():
    games = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "home_win": [1, 0, 1],
            "margin": [7, -3, 10],
            "market_spread": [3.0, -1.0, np.nan],
            "market_prob_home": [0.6, 0.45, np.nan],
            "model_prob": [0.7, 0.4, 0.9],
            "model_spread": [5.0, -2.0, 8.0],
        }
    )

    table = evaluate.score_seasons(games, "model_prob", "model_spread")

    assert table.loc["all", "games"] == 2
    assert table.loc["all", "model_mae"] == pytest.approx((2 + 1) / 2)
    assert table.loc["all", "market_mae"] == pytest.approx((4 + 2) / 2)
