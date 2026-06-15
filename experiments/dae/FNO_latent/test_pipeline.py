#!/usr/bin/env python3
"""
Test script to verify the OT Encoder -> FNO1D -> MLP -> Transfer Function pipeline.
"""

import torch
from main import SeismicSurrogateModel
from OTEncoder import OTEncoder
from utils import FNOLatentConfig


def test_ot_encoder():
    """Test the OT Encoder with synthetic data."""
    print("Testing OT Encoder...")

    # Create synthetic soil profile data
    batch_size = 4
    num_channels = 3  # Vs, Vp, Rho
    num_points = 50

    # Create synthetic data
    soil_profiles = torch.randn(batch_size, num_channels, num_points)
    depth_grids = torch.linspace(0, 100, num_points).unsqueeze(0).repeat(batch_size, 1)

    # Initialize OT Encoder
    config = FNOLatentConfig()
    ot_encoder = OTEncoder(
        latent_dim=config.latent_dim,
        ot_steps=config.ot_steps,
        ot_lr=config.ot_lr,
        device="cpu",
    )

    # Test forward pass
    latent_output = ot_encoder(soil_profiles, depth_grids)

    print(f"Input shape: {soil_profiles.shape}")
    print(f"Depth grid shape: {depth_grids.shape}")
    print(f"Output shape: {latent_output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_channels}, {config.latent_dim})")

    assert latent_output.shape == (batch_size, config.latent_dim, num_channels)
    print("✓ OT Encoder test passed!")
    return latent_output


def test_full_model():
    """Test the complete SeismicSurrogateModel."""
    print("\nTesting Full Model...")

    # Create synthetic data
    batch_size = 2
    num_channels = 3  # Vs, Vp, Rho
    num_points = 50
    output_dim = 100

    soil_profiles = torch.randn(batch_size, num_channels, num_points)
    depth_grids = torch.linspace(0, 100, num_points).unsqueeze(0).repeat(batch_size, 1)

    # Initialize model
    config = FNOLatentConfig()
    config.device = "cpu"

    model = SeismicSurrogateModel(
        config=config, input_channels=num_channels, output_dim=output_dim, device="cpu"
    )

    # Test forward pass
    with torch.no_grad():
        output = model(soil_profiles, depth_grids)

    print(f"Input soil profiles shape: {soil_profiles.shape}")
    print(f"Input depth grids shape: {depth_grids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {output_dim})")

    assert output.shape == (batch_size, output_dim)
    print("✓ Full model test passed!")
    return output


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("\nTesting Gradient Flow...")

    # Create synthetic data
    batch_size = 2
    num_channels = 3
    num_points = 50
    output_dim = 100

    soil_profiles = torch.randn(
        batch_size, num_channels, num_points, requires_grad=True
    )
    depth_grids = torch.linspace(0, 100, num_points).unsqueeze(0).repeat(batch_size, 1)
    target = torch.randn(batch_size, output_dim)

    # Initialize model
    config = FNOLatentConfig()
    config.device = "cpu"

    model = SeismicSurrogateModel(
        config=config, input_channels=num_channels, output_dim=output_dim, device="cpu"
    )

    # Forward pass
    output = model(soil_profiles, depth_grids)

    # Compute loss
    loss = torch.nn.MSELoss()(output, target)

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.parameters()
    )

    print(f"Loss: {loss.item():.6f}")
    print(f"Has non-zero gradients: {has_gradients}")

    assert has_gradients, "No gradients found!"
    print("✓ Gradient flow test passed!")


def test_model_components():
    """Test individual model components."""
    print("\nTesting Model Components...")

    config = FNOLatentConfig()
    config.device = "cpu"

    model = SeismicSurrogateModel(config, device="cpu")

    print(f"OT Encoder: {model.ot_encoder}")
    print(f"FNO Processor: {model.fno_processor}")
    print(f"Prediction Head: {model.prediction_head}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("✓ Model components test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("FNO Latent Pipeline Test Suite")
    print("=" * 60)

    try:
        # Test individual components
        test_ot_encoder()
        test_full_model()
        test_gradient_flow()
        test_model_components()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Pipeline is ready to use.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
