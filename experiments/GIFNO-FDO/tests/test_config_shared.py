"""Ensure FDO config exposes constants required by shared GIFNO modules."""

import config


def test_data_loader_grid_constants():
    assert config.X_SLICE_START == config.BC_WIDTH
    assert config.X_SLICE_END == config.BC_WIDTH + config.LX_VARIABILITY
    assert config.NX == config.LX_VARIABILITY
    assert config.X_SLICE_END - config.X_SLICE_START == config.NX
