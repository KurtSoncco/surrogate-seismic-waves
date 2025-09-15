import pytest
import torch

from wave_surrogate.models.fno.model import (
    Encoder,
    EncoderOperatorModel,
    OperatorDecoder,
)

# --- Test Configuration ---

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


class TestEncoderOperatorModel:
    """Groups tests for the end-to-end EncoderOperatorModel."""

    @pytest.fixture(scope="class")
    def model_and_data(self):
        """
        A fixture to build the full model and create sample data once per test class.
        This is efficient as model instantiation can be slow.
        """
        # Define the full model configuration
        config = {
            "batch_size": 4,
            "input_seq_len": 256,
            "output_seq_len": 128,
            "encoder_channels": [1, 16, 32, 64],
            "latent_dim": 32,
            "fno_modes": 16,
            "fno_width": 64,
            "num_fno_layers": 4,
        }

        # 1. Initialize the Encoder
        encoder = Encoder(
            channels=config["encoder_channels"],
            latent_dim=config["latent_dim"],
        )

        # 2. Initialize the OperatorDecoder
        decoder = OperatorDecoder(
            latent_dim=config["latent_dim"],
            output_size=config["output_seq_len"],
            fno_modes=config["fno_modes"],
            fno_width=config["fno_width"],
            num_fno_layers=config["num_fno_layers"],
        )

        # 3. Combine them in the main model
        model = EncoderOperatorModel(encoder=encoder, decoder=decoder)

        # 4. Create sample input data
        x = torch.randn(config["batch_size"], config["input_seq_len"])

        return model, x, config

    @pytest.mark.parametrize("device", DEVICES)
    def test_model_forward_pass_and_shape(self, model_and_data, device):
        """
        Tests the end-to-end forward pass and verifies the final output shape.
        """
        # Arrange
        model, x, config = model_and_data
        model.to(device)
        x = x.to(device)

        # Act
        output = model(x)

        # Assert
        expected_shape = (config["batch_size"], config["output_seq_len"])
        assert output.shape == expected_shape
        assert output.device == x.device
        assert output.dtype == x.dtype

    @pytest.mark.parametrize("device", DEVICES)
    def test_model_backward_pass(self, model_and_data, device):
        """
        Ensures the entire computational graph is differentiable and learnable.
        """
        # Arrange
        model, x, config = model_and_data
        model.to(device)
        x = x.to(device)

        # Act
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Assert
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient."
