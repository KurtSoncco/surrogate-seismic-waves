from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = np.load(Path(__file__).parent / "pce_antiderivative_results.npz")
x = data["x"]
s_true = data["s_true"]
s_dd = data["s_pred_data_driven"]
s_pi = data["s_pred_physics"]

# Set a random seed for reproducibility
rng = np.random.default_rng(42)

# Get the number of samples
num_samples = s_true.shape[0]

# Choose 3 random sample indices
num_subplots = 3
random_indices = rng.choice(num_samples, size=num_subplots, replace=False)

# Plot the random examples
fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 12), sharex=True)

for i, sample_idx in enumerate(random_indices):
    ax = axes[i]
    ax.plot(x, s_true[sample_idx, :], "k-", label="Ground Truth")
    ax.plot(x, s_dd[sample_idx, :], "b--", label="Data-Driven PCE")
    ax.plot(
        x,
        s_pi[sample_idx, :],
        "r:",
        label="Physics-Informed PCE",
    )
    ax.set_title(f"Example Realization {sample_idx + 1}")
    ax.set_ylabel("s(x)")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("x")
fig.suptitle("Anti-Derivative Problem: PCE Predictions vs Ground Truth", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(Path(__file__).parent / "pce_antiderivative_examples.png")
plt.close(fig)
