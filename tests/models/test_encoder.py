import pytest
import torch

from wave_surrogate.models.fno.model import Encoder

# --- Test Configuration ---

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


class TestEncoder:
    """Groups all tests for the configurable Encoder."""

    @pytest.fixture
    def encoder_config(self):
        """Provides a reusable configuration for a standard encoder."""
        return {
            "batch_size": 8,
            "seq_length": 256,
            "channels": [1, 32, 64, 128],
            "latent_dim": 10,
        }

    @pytest.mark.parametrize("device", DEVICES)
    def test_encoder_forward_and_shape(self, encoder_config, device):
        """
        Tests the forward pass and verifies the output tensor's shape.
        """
        # Arrange
        model = Encoder(
            channels=encoder_config["channels"],
            latent_dim=encoder_config["latent_dim"],
        ).to(device)
        x = torch.randn(encoder_config["batch_size"], encoder_config["seq_length"]).to(
            device
        )

        # Act
        y = model(x)

        # Assert
        assert y.shape == (encoder_config["batch_size"], encoder_config["latent_dim"])
        assert y.device == x.device
        assert y.dtype == x.dtype

    @pytest.mark.parametrize("device", DEVICES)
    def test_encoder_backward_pass(self, encoder_config, device):
        """
        Checks for successful backpropagation to ensure the model is learnable.
        """
        # Arrange
        model = Encoder(
            channels=encoder_config["channels"],
            latent_dim=encoder_config["latent_dim"],
        ).to(device)
        x = torch.randn(encoder_config["batch_size"], encoder_config["seq_length"]).to(
            device
        )

        # Act
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Assert
        for name, param in model.named_parameters():
            # LazyLinear params might not be initialized if forward not called
            if "fc" in name and y.numel() == 0:
                continue
            assert param.grad is not None, f"Parameter '{name}' has no gradient."

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "test_channels",
        [
            [1, 16, 32],  # Shallow network
            [1, 32, 64, 128, 256],  # Deeper network
        ],
    )
    def test_encoder_different_configs(self, encoder_config, test_channels, device):
        """
        Tests that the encoder can be initialized and run with various channel configs.
        """
        # Arrange
        latent_dim = encoder_config["latent_dim"]
        model = Encoder(channels=test_channels, latent_dim=latent_dim).to(device)
        x = torch.randn(encoder_config["batch_size"], encoder_config["seq_length"]).to(
            device
        )

        # Act
        try:
            y = model(x)
        except Exception as e:
            pytest.fail(f"Model failed with channels={test_channels}. Error: {e}")

        # Assert
        assert y.shape == (encoder_config["batch_size"], latent_dim)
