from football_model import config


def test_paths_point_inside_repo():
    for path in [config.RAW_DIR, config.PREDICTIONS_DIR, config.FIGURES_DIR]:
        assert config.ROOT in path.parents


def test_season_range_is_sensible():
    assert config.FIRST_SEASON < config.COVID_SEASON < config.CURRENT_SEASON
