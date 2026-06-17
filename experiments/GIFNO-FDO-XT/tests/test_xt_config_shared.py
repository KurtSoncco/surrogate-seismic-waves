"""Ensure XT config exposes constants required by shared GIFNO modules."""

import config


def test_data_loader_grid_constants():
    assert config.X_SLICE_START == config.BC_WIDTH
    assert config.X_SLICE_END == config.BC_WIDTH + config.LX_VARIABILITY
    assert config.NX == config.LX_VARIABILITY
    assert config.X_SLICE_END - config.X_SLICE_START == config.NX


def test_winner_defaults():
    assert config.DEEPONET_LATENT_DIM == 128
    assert config.LOSS_P == 1
    assert config.AMSGRAD is True
    assert config.LOSS_H1_WEIGHT == 0.25
    assert config.X_COORD_MODE == "normalized"
    assert config.WANDB_PROJECT == "gifno_fdo_xt"
