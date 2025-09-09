from pathlib import Path

import matplotlib
import numpy as np
import pytest

from wave_surrogate.plot_utils.radar_graph import _normalize_data, radar_graph

matplotlib.use("Agg")


# --- Test Data Fixture ---
@pytest.fixture
def sample_data():
    """Provides a consistent set of data for tests."""
    return {
        "Model A": {"MSE": 0.1, "MAE": 0.2, "R2": 0.9},
        "Model B": {"MSE": 0.3, "MAE": 0.1, "R2": 0.8},
    }


# --- Test Cases ---
def test_plot_properties(sample_data):
    """
    Test 1: Verify that the generated plot has the correct visual components.
    """
    title = "Test Radar Chart"
    fig, ax = radar_graph(sample_data, title=title, show_plot=False)  # type: ignore

    # 1. Check if the title is set correctly
    assert ax.title.get_text() == title

    # 2. Check if the correct number of models are plotted (one filled polygon per model)
    # The `fill` method creates `Polygon` patches, not collections.
    assert len(ax.patches) == len(sample_data)

    # 3. Check if the legend contains the correct model names
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert sorted(legend_texts) == sorted(sample_data.keys())

    # 4. Check if the correct number of axes (metrics) are created
    num_metrics = len(next(iter(sample_data.values())))
    assert len(ax.get_xticks()) == num_metrics, "Incorrect number of metric axes."


def test_insufficient_metrics_error():
    """
    Test 2: Ensure the function raises a ValueError if fewer than 3 metrics are provided.
    """
    invalid_data = {"Model A": {"MSE": 0.1, "R2": 0.9}}  # Only 2 metrics
    with pytest.raises(ValueError, match="Radar charts require at least 3 metrics."):
        radar_graph(invalid_data, show_plot=False)


def test_plot_saving(sample_data, tmp_path: Path):
    """
    Test 3: Verify that the plot is saved to a file when save_path is provided.

    Uses pytest's built-in `tmp_path` fixture to create a temporary directory.
    """
    save_file = tmp_path / "test_chart.png"
    radar_graph(sample_data, save_path=str(save_file), show_plot=False)

    assert save_file.is_file(), "The plot file was not created at the specified path."


def test_normalization_logic(sample_data):
    """
    Test 4: Directly test the _normalize_data helper function for correctness.
    This is a pure unit test on the data transformation logic.
    """
    labels = ["MSE", "MAE", "R2"]
    lower_is_better = ["MSE", "MAE"]

    normalized_data, _, _ = _normalize_data(sample_data, labels, lower_is_better)

    # Expected values:
    # MSE (lower is better): min=0.1, max=0.3. Model A (0.1) -> (0.3-0.1)/(0.3-0.1) = 1.0. Model B (0.3) -> (0.3-0.3)/(0.3-0.1) = 0.0.
    # MAE (lower is better): min=0.1, max=0.2. Model A (0.2) -> (0.2-0.2)/(0.2-0.1) = 0.0. Model B (0.1) -> (0.2-0.1)/(0.2-0.1) = 1.0.
    # R2 (higher is better): min=0.8, max=0.9. Model A (0.9) -> (0.9-0.8)/(0.9-0.8) = 1.0. Model B (0.8) -> (0.8-0.8)/(0.9-0.8) = 0.0.
    expected = np.array(
        [
            [1.0, 0.0, 1.0],  # Model A
            [0.0, 1.0, 0.0],  # Model B
        ]
    )

    # Use numpy's testing utility for robust floating-point comparison
    np.testing.assert_allclose(normalized_data, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
