import pytest
import torch

from wave_surrogate.models.dae.architectures import Decoder, Encoder, WeakerDecoder
from wave_surrogate.models.dae.base import DecoupledAutoencoder


@pytest.fixture
def sample_input():
    return torch.randn(10, 31)  # Batch of 10 samples, each of dimension 31


def test_encoder_forward(sample_input):
    encoder = Encoder(input_dim=31, hidden_dim=[64, 32], latent_dim=16)
    z = encoder(sample_input)
    assert z.shape == (10, 16), "Encoder output shape mismatch"


def test_decoder_forward():
    decoder = Decoder(latent_dim=16, hidden_dim=[32, 64], output_dim=31)
    z = torch.randn(10, 16)
    x_hat = decoder(z)
    assert x_hat.shape == (10, 31), "Decoder output shape mismatch"


def test_weaker_decoder_forward():
    weaker_decoder = WeakerDecoder(
        latent_dim=16, hidden_dim=[32, 64], output_dim=31, dropout_rate=0.5
    )
    z = torch.randn(10, 16)
    x_hat = weaker_decoder(z)
    assert x_hat.shape == (10, 31), "Weaker Decoder output shape mismatch"


def test_dae_forward(sample_input):
    encoder = Encoder(input_dim=31, hidden_dim=[64, 32], latent_dim=16)
    decoder = Decoder(latent_dim=16, hidden_dim=[32, 64], output_dim=31)
    weaker_decoder = WeakerDecoder(
        latent_dim=16, hidden_dim=[32, 64], output_dim=31, dropout_rate=0.5
    )

    dae = DecoupledAutoencoder(
        encoder=encoder, decoder=decoder, aux_decoder=weaker_decoder
    )

    # Test Stage 1 forward
    x_hat_stage1, z_stage1 = dae.forward_stage1(sample_input)
    assert x_hat_stage1.shape == (10, 31), "DAE Stage 1 output shape mismatch"
    assert z_stage1.shape == (10, 16), "DAE Stage 1 latent shape mismatch"

    # Test standard forward
    x_hat, z = dae.forward(sample_input)
    assert x_hat.shape == (10, 31), "DAE output shape mismatch"
    assert z.shape == (10, 16), "DAE latent shape mismatch"
