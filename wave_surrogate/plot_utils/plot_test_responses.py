import matplotlib.pyplot as plt
import seaborn as sns

from wave_surrogate.logging_setup import setup_logging

logger = setup_logging()

sns.set_palette("colorblind")


def plot_test_responses(
    true_responses, pred_responses, freq_data, save_path=None, n_columns=4, n_rows=4
):
    """
    Plots true vs predicted test responses.

    Args:
        true_responses (np.ndarray): True test responses.
        pred_responses (np.ndarray): Predicted test responses.
        freq_data (np.ndarray): Frequency data for x-axis.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
        n_columns (int): Number of columns in the subplot grid. Default is 4.
        n_rows (int): Number of rows in the subplot grid. Default is 4.
    """
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(20, 15))
    for i in range(n_rows):
        for j in range(n_columns):
            ax[i, j].plot(
                freq_data,
                true_responses.T,
                label="True Responses",
                color="blue",
                alpha=0.5,
            )
            ax[i, j].plot(
                freq_data,
                pred_responses.T,
                label="Predicted Responses",
                color="orange",
                alpha=0.5,
            )
    plt.xlabel("Frequency")
    plt.ylabel("Response")
    plt.title("True vs Predicted Test Responses")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
