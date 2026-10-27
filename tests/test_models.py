import numpy as np
import pandas as pd
import pytest

from football_model import config, models


def make_features(seasons: list[int], n_per_season: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic games where margin = 3 * x + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 1
    for season in seasons:
        x = rng.normal(size=n_per_season)
        margin = 3 * x + rng.normal(scale=1.0, size=n_per_season)
        for xi, mi in zip(x, margin):
            rows.append({"game_id": game_id, "season": season, "x": xi, "margin": mi})
            game_id += 1
    return pd.DataFrame(rows)


def test_training_rows_are_earlier_seasons_minus_exclusions():
    feats = make_features([2014, 2015, 2019, 2020, 2021, 2022])

    train = models.training_rows(feats, 2022)

    assert sorted(train["season"].unique()) == [2015, 2019, 2021]
    assert config.BURN_IN_SEASON == 2014 and config.COVID_SEASON == 2020


def test_walk_forward_never_trains_on_the_predicted_season():
    feats = make_features([2015, 2016, 2017])
    pred_before = models.walk_forward_stacked(feats, ["x"], 1.0, [2017])

    # Wildly change 2017's outcomes: its own predictions must not move.
    changed = feats.copy()
    changed.loc[changed["season"] == 2017, "margin"] *= -10
    pred_after = models.walk_forward_stacked(changed, ["x"], 1.0, [2017])

    pd.testing.assert_frame_equal(pred_before, pred_after)


def test_scaling_uses_training_rows_only():
    feats = make_features([2015, 2016])
    model, _ = models.fit_stacked(feats[feats["season"] == 2015], ["x"], 1.0)
    train_x = feats.loc[feats["season"] == 2015, "x"]

    assert model.named_steps["scale"].mean_[0] == pytest.approx(train_x.mean())


def test_stacked_model_recovers_slope_and_probabilities():
    feats = make_features([2015, 2016, 2017], n_per_season=500)

    model, sigma = models.fit_stacked(feats, ["x"], alpha=0.0)
    pred = models.walk_forward_stacked(feats, ["x"], 0.0, [2017])

    # Coefficient on standardized x is 3 * sd(x); back on x's scale it is about 3.
    slope = model.named_steps["ridge"].coef_[0] / model.named_steps["scale"].scale_[0]
    assert slope == pytest.approx(3.0, abs=0.15)
    assert sigma == pytest.approx(1.0, abs=0.1)
    assert ((pred["stack_prob"] > 0.5) == (pred["stack_spread"] > 0)).all()


def test_coefficient_table_has_standard_errors():
    feats = make_features([2015], n_per_season=300)
    table = models.coefficient_table(feats, ["x"])

    assert list(table.columns) == ["coef", "std_err", "t", "p_value"]
    assert table.loc["x", "coef"] == pytest.approx(3 * feats["x"].std(), rel=0.1)
