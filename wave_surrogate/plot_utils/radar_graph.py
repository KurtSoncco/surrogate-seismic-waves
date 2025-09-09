from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set a visually appealing and accessible style
sns.set_theme(style="whitegrid", palette="colorblind")


def _normalize_data(
    data: Dict[str, Dict[str, float]],
    labels: List[str],
    lower_is_better_metrics: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes model performance data using NumPy for vectorization.

    Returns:
        - A NumPy array of the normalized data.
        - A NumPy array of the minimum values for each metric.
        - A NumPy array of the maximum values for each metric.
    """
    # Create a 2D NumPy array (models x metrics)
    model_names = list(data.keys())
    raw_values = np.array(
        [[data[model][metric] for metric in labels] for model in model_names]
    )

    # Find min and max for each metric (column-wise)
    min_vals = np.min(raw_values, axis=0)
    max_vals = np.max(raw_values, axis=0)

    # Calculate the range, handling the case where min == max
    data_range = max_vals - min_vals
    data_range[data_range == 0] = 1.0  # Avoid division by zero

    # Vectorized normalization
    normalized_values = (raw_values - min_vals) / data_range

    # Invert scores for 'lower is better' metrics
    for i, metric in enumerate(labels):
        if metric in lower_is_better_metrics:
            normalized_values[:, i] = 1 - normalized_values[:, i]

    return normalized_values, min_vals, max_vals


def radar_graph(
    data: Dict[str, Dict[str, float]],
    lower_is_better_metrics: List[str] = ["RMSE", "MAE", "MSE", "MAPE"],
    title: str = "Radar Chart",
    figsize: Tuple[int, int] = (10, 10),
    save_path: str | None = None,
    num_grid_lines: int = 5,
    show_plot: bool = True,
):
    """
    Generates a radar chart with normalized axes and dynamic, original-scale data labels.

    This function normalizes each metric to a 0-1 scale for fair comparison, but displays
    the original scale on each axis for better interpretability.

    Args:
        data (Dict[str, Dict[str, float]]):
            A nested dictionary of data. Outer keys are group names (e.g., 'Model A'),
            and inner keys are metric names with their corresponding values.
        lower_is_better_metrics (List[str], optional):
            A list of metric names where a lower value is better (e.g., ['RMSE', 'MAE']).
            These will be inverted during normalization so that "better" is always outward.
        title (str): The title for the chart.
        figsize (tuple): The width and height of the figure.
        save_path (str | None): If provided, the file path to save the plot image.
        num_grid_lines (int): The number of concentric grid lines to display.
    """
    if not data:
        print("Warning: The data dictionary is empty. Nothing to plot.")
        return

    # --- Setup ---
    model_names = list(data.keys())
    labels = list(data[model_names[0]].keys())
    num_vars = len(labels)

    if num_vars < 3:
        raise ValueError("Radar charts require at least 3 metrics.")

    # Get a color for each model
    colors = sns.color_palette("colorblind", n_colors=len(data))

    # Calculate angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # --- Data Normalization ---
    normalized_data, min_vals, max_vals = _normalize_data(
        data, labels, lower_is_better_metrics
    )

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot each model's data
    for i, model_name in enumerate(model_names):
        values = np.concatenate((normalized_data[i], [normalized_data[i, 0]]))
        ax.plot(angles, values, label=model_name, color=colors[i], linewidth=2)
        ax.fill(angles, values, color=colors[i], alpha=0.2)

    # --- Formatting & Custom Axis Labels ---
    ax.set_title(title, size=20, color="black", y=1.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Hide default labels, we'll draw our own

    # Set the y-axis limit slightly larger to avoid labels being cut off
    ax.set_ylim(0, 1.05)
    # Hide the default radial axis labels
    ax.set_yticklabels([])
    # Turn off the default grid
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # First, create a fine-grained set of angles for a smooth circle
    circle_angles = np.linspace(0, 2 * np.pi, 100)

    # Get the grid levels as before
    grid_levels = np.linspace(0, 1, num=num_grid_lines)

    # Now, plot a smooth circle for each level
    for level in grid_levels[1:]:  # We skip level 0, which is just the center point
        ax.plot(
            circle_angles,
            [level] * 100,
            color="gray",
            linestyle=":",
            linewidth=1,
            zorder=-2,
        )

    # Draw custom axis labels (metric names and their scales)
    for i, (label, angle) in enumerate(zip(labels, angles[:-1])):
        # Metric Names (positioned at the outer edge)
        ax.text(
            angle,
            1.1,  # Position slightly outside the max grid line
            label,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
        )

        # Metric Scales
        is_lower_better = label in lower_is_better_metrics
        for level in grid_levels[1:]:  # Skip the center point (0)
            # Determine the original value that corresponds to this grid level
            if is_lower_better:
                # For inverted axes, the inner ring is max_val, outer is min_val
                original_val = max_vals[i] - level * (max_vals[i] - min_vals[i])
            else:
                original_val = min_vals[i] + level * (max_vals[i] - min_vals[i])

            # Format the label string
            label_text = (
                f"{original_val:.2f}"
                if isinstance(original_val, float)
                else str(original_val)
            )

            # Place the text label along the axis
            ax.text(
                angle,
                level,
                label_text,
                ha="center",
                va="center",
                fontsize=9,
                color="dimgray",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.1),
            )

    # --- Legend and Final Touches ---
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=12)
    plt.tight_layout(pad=3)  # Add padding to ensure title and labels fit

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    elif show_plot:
        plt.show()

    return fig, ax
