#!/usr/bin/env python3
"""
Quick test to verify the OT Encoder fixes.
"""

import torch
from OTEncoder import OTEncoder


def test_ot_encoder_fix():
    """Test the OT Encoder with the fixes."""
    print("Testing OT Encoder fixes...")

    # Create synthetic data
    batch_size = 2
    num_channels = 3
    num_points = 20

    soil_profiles = torch.randn(batch_size, num_channels, num_points)
    depth_grids = torch.linspace(0, 100, num_points).unsqueeze(0).repeat(batch_size, 1)

    # Initialize OT Encoder
    ot_encoder = OTEncoder(
        latent_dim=16,
        ot_steps=5,  # Reduced for faster testing
        ot_lr=0.1,
        device="cpu",
    )

    try:
        # Test forward pass
        output = ot_encoder(soil_profiles, depth_grids)
        print("✓ OT Encoder test passed!")
        print(f"  Input shape: {soil_profiles.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, {num_channels}, 16)")
        return True
    except Exception as e:
        print(f"❌ OT Encoder test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ot_encoder_fix()
    if success:
        print("\n🎉 All fixes working correctly!")
    else:
        print("\n❌ Issues remain to be fixed.")
