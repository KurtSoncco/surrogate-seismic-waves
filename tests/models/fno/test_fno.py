import pytest
import torch

from wave_surrogate.models.fno.model import FNO1D

# --- Test Configuration ---

# Automatically select devices to test on (CPU and CUDA if available)
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


class TestFNO1D:
    """Groups all tests for the FNO1D layer."""

    @pytest.fixture
    def fno_config(self):
        """Provides a reusable configuration dictionary for the FNO model."""
        return {
            "batch_size": 4,
            "seq_length": 100,
            "width": 64,
            "modes": 16,
        }

    @pytest.mark.parametrize("device", DEVICES)
    def test_fno1d_forward_and_shape(self, fno_config, device):
        """
        Tests the forward pass and verifies the output tensor's shape and device.
        """
        # Arrange: Create model and input tensor on the correct device
        model = FNO1D(modes=fno_config["modes"], width=fno_config["width"]).to(device)
        x = torch.randn(
            fno_config["batch_size"],
            fno_config["seq_length"],
            fno_config["width"],
        ).to(device)

        # Act: Perform forward pass
        y = model(x)

        # Assert: Check shape, dtype, and device
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert y.device == x.device

    @pytest.mark.parametrize("device", DEVICES)
    def test_fno1d_backward_pass(self, fno_config, device):
        """
        Checks for successful backpropagation to ensure the model is learnable.
        """
        # Arrange
        model = FNO1D(modes=fno_config["modes"], width=fno_config["width"]).to(device)
        x = torch.randn(
            fno_config["batch_size"],
            fno_config["seq_length"],
            fno_config["width"],
        ).to(device)

        # Act: Simulate a training step
        y = model(x)
        loss = y.sum()  # A dummy loss
        loss.backward()

        # Assert: Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient."

    @pytest.mark.parametrize("device", DEVICES)
    def test_fno1d_modes_edge_case(self, fno_config, device):
        """
        Tests that the model handles cases where modes > seq_length // 2.
        The model should internally cap the modes used.
        """
        # Arrange: Set modes to an intentionally high value
        high_modes = fno_config["seq_length"]  # Guaranteed to be > seq_length / 2
        model = FNO1D(modes=high_modes, width=fno_config["width"]).to(device)
        x = torch.randn(
            fno_config["batch_size"],
            fno_config["seq_length"],
            fno_config["width"],
        ).to(device)

        # Act & Assert: Ensure forward pass completes without error
        try:
            y = model(x)
            # Check that output shape is still correct
            assert y.shape == x.shape
        except Exception as e:
            pytest.fail(f"Model failed on modes edge case with error: {e}")
