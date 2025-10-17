# test_comprehensive.py
"""
Comprehensive test suite for the latent FNO pipeline.
Tests functionality, metrics, and different configurations.
"""

import os
import sys

import numpy as np
import torch


def _ensure_local_src_on_path():
    test_dir = os.path.abspath(os.path.dirname(__file__))
    cur = test_dir
    for _ in range(6):  # search up to 6 levels
        candidate_src = os.path.join(cur, "src")
        if os.path.isdir(candidate_src):
            # add `cur` so `import src...` resolves to cur/src
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: also try adding experiments/latent_FNO top folder as fallback
    fallback = os.path.abspath(os.path.join(test_dir, ".."))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_ensure_local_src_on_path()


from src.configs.config import get_config
from src.models.pipeline import (
    AblationStudyPipeline,
    create_pipeline,
)
from src.utils.data_utils import DataAugmentation
from src.utils.metrics import MetricsCalculator, ModelEvaluator
from src.utils.wandb_utils import WandbLogger


def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("Testing basic pipeline functionality...")

    # Create a simple pipeline
    model = create_pipeline(
        input_dim=29,
        output_dim=1000,
        latent_dim=128,
        encoder_type="mlp",
        decoder_type="mlp",
        fno_processor_type="simple",
    )

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 29)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, 1000), (
        f"Expected output shape (16, 1000), got {output.shape}"
    )

    # Test gradient flow
    x.requires_grad_(True)
    y = torch.randn(batch_size, 1000)
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    assert x.grad is not None, "Input gradients not computed"

    print("✓ Basic functionality test passed")


def test_all_encoders():
    """Test all encoder types."""
    print("Testing all encoder types...")

    encoders = ["mlp", "cnn", "transformer"]
    x = torch.randn(8, 29)

    for encoder_type in encoders:
        model = create_pipeline(
            input_dim=29,
            output_dim=1000,
            latent_dim=128,
            encoder_type=encoder_type,
            decoder_type="mlp",
            fno_processor_type="simple",
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (8, 1000), f"Encoder {encoder_type} failed"
        print(f"✓ {encoder_type} encoder test passed")


def test_all_fno_processors():
    """Test all FNO processor types."""
    print("Testing all FNO processor types...")

    processors = ["simple", "sequence", "multiscale", "adaptive"]
    x = torch.randn(8, 29)

    for processor_type in processors:
        model = create_pipeline(
            input_dim=29,
            output_dim=1000,
            latent_dim=128,
            encoder_type="mlp",
            decoder_type="mlp",
            fno_processor_type=processor_type,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (8, 1000), f"FNO processor {processor_type} failed"
        print(f"✓ {processor_type} FNO processor test passed")


def test_all_decoders():
    """Test all decoder types."""
    print("Testing all decoder types...")

    decoders = ["mlp", "cnn", "transformer", "fno_operator"]
    x = torch.randn(8, 29)

    for decoder_type in decoders:
        model = create_pipeline(
            input_dim=29,
            output_dim=1000,
            latent_dim=128,
            encoder_type="mlp",
            decoder_type=decoder_type,
            fno_processor_type="simple",
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (8, 1000), f"Decoder {decoder_type} failed"
        print(f"✓ {decoder_type} decoder test passed")


def test_ablation_study():
    """Test ablation study functionality."""
    print("Testing ablation study...")

    # Full pipeline
    full_model = AblationStudyPipeline(
        input_dim=29,
        output_dim=1000,
        latent_dim=128,
        use_encoder=True,
        use_fno=True,
        use_decoder=True,
    )

    # No FNO
    no_fno_model = AblationStudyPipeline(
        input_dim=29,
        output_dim=1000,
        latent_dim=128,
        use_encoder=True,
        use_fno=False,
        use_decoder=True,
    )

    x = torch.randn(8, 29)

    with torch.no_grad():
        full_output = full_model(x)
        no_fno_output = no_fno_model(x)

    assert full_output.shape == (8, 1000), "Full pipeline failed"
    assert no_fno_output.shape == (8, 1000), "No FNO pipeline failed"

    print("✓ Ablation study test passed")


def test_configurations():
    """Test predefined configurations."""
    print("Testing predefined configurations...")

    configs_to_test = ["baseline", "cnn_encoder", "sequence_fno", "multiscale_fno"]

    for config_name in configs_to_test:
        config = get_config(config_name)

        model = create_pipeline(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            latent_dim=config.latent_dim,
            encoder_type=config.encoder_type,
            decoder_type=config.decoder_type,
            fno_processor_type=config.fno_processor_type,
            encoder_config=config.encoder_config,
            decoder_config=config.decoder_config,
            fno_processor_config=config.fno_processor_config,
        )

        x = torch.randn(8, config.input_dim)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (8, config.output_dim), f"Config {config_name} failed"
        print(f"✓ {config_name} config test passed")


def test_metrics_calculator():
    """Test metrics calculator."""
    print("Testing metrics calculator...")

    # Create dummy data
    batch_size = 32
    output_dim = 100

    predictions = torch.randn(batch_size, output_dim)
    targets = predictions + torch.randn(batch_size, output_dim) * 0.1  # Add some noise

    # Create frequency data
    frequencies = np.linspace(0.1, 10.0, output_dim)

    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(predictions, targets, frequencies)

    # Check that all expected metrics are present
    expected_metrics = [
        "mse",
        "rmse",
        "mae",
        "mape",
        "r2",
        "explained_variance",
        "pearson_correlation",
        "spearman_correlation",
        "kendall_tau",
        "mean_sample_correlation",
        "mean_frequency_correlation",
        "residual_mean",
        "residual_std",
        "residual_skewness",
        "residual_kurtosis",
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert not np.isnan(metrics[metric]), f"Metric {metric} is NaN"

    print("✓ Metrics calculator test passed")


def test_data_augmentation():
    """Test data augmentation utilities."""
    print("Testing data augmentation...")

    # Create dummy data
    data = torch.randn(16, 100)

    # Test noise addition
    noisy_data = DataAugmentation.add_noise(data, noise_level=0.01)
    assert noisy_data.shape == data.shape, "Noise augmentation failed"

    # Test magnitude scaling
    scaled_data = DataAugmentation.magnitude_scaling(data, scale_range=(0.8, 1.2))
    assert scaled_data.shape == data.shape, "Magnitude scaling failed"

    print("✓ Data augmentation test passed")


def test_model_evaluator():
    """Test model evaluator."""
    print("Testing model evaluator...")

    # Create a simple model
    model = create_pipeline(
        input_dim=29,
        output_dim=100,
        latent_dim=64,
        encoder_type="mlp",
        decoder_type="mlp",
        fno_processor_type="simple",
    )

    # Create dummy dataset
    batch_size = 16
    x = torch.randn(batch_size, 29)
    y = torch.randn(batch_size, 100)

    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Create frequency data
    frequencies = np.linspace(0.1, 10.0, 100)

    # Evaluate model
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, dataloader, frequencies)

    # Check results
    assert "metrics" in results, "Missing metrics in evaluation results"
    assert "predictions" in results, "Missing predictions in evaluation results"
    assert "targets" in results, "Missing targets in evaluation results"

    assert results["predictions"].shape == y.shape, "Prediction shape mismatch"
    assert results["targets"].shape == y.shape, "Target shape mismatch"

    print("✓ Model evaluator test passed")


def test_wandb_logger():
    """Test W&B logger (without actual logging)."""
    print("Testing W&B logger...")

    # Create logger with logging disabled
    logger = WandbLogger(
        project="test-project",
        name="test-run",
        config={"test_param": 42},
        enabled=False,  # Disable actual W&B logging
    )

    # Test basic functionality
    logger.log_metrics({"test_metric": 0.5})
    logger.log_model_parameters(torch.nn.Linear(10, 1))

    # Test with dummy data
    predictions = torch.randn(10, 50)
    targets = torch.randn(10, 50)

    logger.log_predictions(predictions, targets, max_samples=5)

    # Test frequency correlation
    frequencies = np.linspace(0.1, 10.0, 50)
    logger.log_correlation_analysis(predictions, targets, frequencies)

    logger.finish()

    print("✓ W&B logger test passed")


def test_performance_comparison():
    """Test performance comparison between different configurations."""
    print("Testing performance comparison...")

    # Create different configurations
    configs = [
        ("baseline", get_config("baseline")),
        ("cnn_encoder", get_config("cnn_encoder")),
        ("sequence_fno", get_config("sequence_fno")),
    ]

    # Create dummy data
    x = torch.randn(32, 29)
    y = torch.randn(32, 1000)

    results = {}

    for config_name, config in configs:
        # Create model
        model = create_pipeline(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            latent_dim=config.latent_dim,
            encoder_type=config.encoder_type,
            decoder_type=config.decoder_type,
            fno_processor_type=config.fno_processor_type,
            encoder_config=config.encoder_config,
            decoder_config=config.decoder_config,
            fno_processor_config=config.fno_processor_config,
        )

        # Test forward pass time
        import time

        # Warm up
        with torch.no_grad():
            _ = model(x)

        # Time the forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(x)
        forward_time = time.time() - start_time

        # Calculate basic metrics
        mse = torch.mean((output - y) ** 2).item()
        mae = torch.mean(torch.abs(output - y)).item()

        results[config_name] = {
            "forward_time": forward_time,
            "mse": mse,
            "mae": mae,
            "parameters": sum(p.numel() for p in model.parameters()),
        }

    # Print comparison
    print("\nPerformance Comparison:")
    print("-" * 60)
    print(f"{'Config':<15} {'Time (ms)':<10} {'MSE':<12} {'MAE':<12} {'Params':<10}")
    print("-" * 60)

    for config_name, result in results.items():
        print(
            f"{config_name:<15} {result['forward_time'] * 1000:<10.2f} "
            f"{result['mse']:<12.6f} {result['mae']:<12.6f} {result['parameters']:<10,}"
        )

    print("✓ Performance comparison test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Comprehensive Test Suite for Latent FNO Pipeline")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_all_encoders,
        test_all_fno_processors,
        test_all_decoders,
        test_ablation_study,
        test_configurations,
        test_metrics_calculator,
        test_data_augmentation,
        test_model_evaluator,
        test_wandb_logger,
        test_performance_comparison,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
