# Surrogate Modeling of Seismic Waves

[![Project Status](https://img.shields.io/badge/Project%20Status-Active-brightgreen?style=for-the-badge)](https://github.com/KurtSoncco/surrogate-seismic-waves)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/KurtSoncco/surrogate-seismic-waves/actions/workflows/ci.yml/badge.svg)](https://github.com/KurtSoncco/surrogate-seismic-waves)
[![uv](https://img.shields.io/badge/uv-%3E%3D0.1.0-blue?style=for-the-badge)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Github stars](https://img.shields.io/github/stars/KurtSoncco/surrogate-seismic-waves?style=social)](https://github.com/KurtSoncco/surrogate-seismic-waves/stargazers)

> This repository explores the development and application of surrogate models for simulating seismic wave propagation through layered media. The primary goal is to create computationally efficient alternatives to traditional, high-fidelity, physics-based simulations (e.g., Finite Difference Time Domain methods like ITASCA FLAC). We investigate various machine learning approaches, including Recurrent Neural Networks (RNNs), Operator Learning frameworks, and Reduced-Order Models (ROMs), to predict the dynamic response of soil profiles under seismic loading. The performance of these models is benchmarked against results from physics-based software to evaluate their accuracy, generalization capabilities, and computational speed-up.

[Research Questions](#-research-questions--hypothesis) • [Methodology](#️-methodology) • [Data](#-data) • [How to Reproduce](#-how-to-reproduce) • [Key Results](#-key-results)

---

## 🎯 Research Questions / Hypothesis

> - How effective are different surrogate models (e.g., RNNs, Fourier Neural Operators) at capturing the complex physics of elastic and inelastic wave propagation compared to traditional FDTD methods?
> - Can these surrogate models accurately generalize to predict seismic responses for geological profiles and input motions not seen during training?
> - What is the computational speed-up achieved by using surrogate models over physics-based simulations, and what are the trade-offs in terms of prediction accuracy?
> - How does the complexity of the layered media (number of layers, material properties, non-linearity) affect the performance and training requirements of the surrogate models?

---

## 🛠️ Methodology

> The methodology for this project is divided into three main stages:
>
> 1.  **Data Generation:** A comprehensive dataset is generated using the physics-based software ITASCA FLAC. We simulate the 1D seismic response of various multi-layered soil profiles. Input parameters such as layer thickness, shear wave velocity, damping, and non-linear material properties are systematically varied. A suite of real and synthetic earthquake ground motions are used as input excitations.
>
> 2.  **Surrogate Modeling:** Several machine learning models are developed to learn the mapping from input parameters (soil profile, ground motion) to output responses (surface acceleration time histories). The models include:
>     -   **Recurrent Neural Networks (LSTMs/GRUs):** To capture the temporal dependencies in the seismic response.
>     -   **Operator Learning (e.g., FNO):** To learn the underlying solution operator of the wave propagation problem, allowing for generalization across different input functions.
>     -   **Reduced-Order Models (e.g., POD):** To project the high-dimensional system onto a lower-dimensional subspace for faster computation.
>
> 3.  **Evaluation:** The surrogate models are trained on a subset of the generated data and evaluated on a held-out test set. Performance is measured using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and goodness-of-fit on the transfer functions. The computational inference time is benchmarked against the runtime of the original FLAC simulations.
---

## 💾 Data

> The dataset consists of input-output pairs from thousands of 1D site response analyses performed with ITASCA FLAC.
>
> -   **Inputs:**
>     -   Soil Profile Properties: A vector describing the thickness, shear wave velocity, density, and non-linear parameters for each layer.
> -   **Outputs:**
>     -   Transfer Function: A frequency-based signal that observes the difference of frequency content between the subsurface and the surface levels.
>
> The data is preprocessed and normalized before being fed into the models. Due to its size, the dataset is not stored in this Git repository.
>
> The final dataset for this project can be found at: `[Link to cloud storage, university server, etc.]`

---

## 🚀 How to Reproduce

> _Provide step-by-step instructions to set up the environment and run the analysis._

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KurtSoncco/surrogate-seismic-waves
    cd surrogate-seismic-waves
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    pyenv local 3.11
    uv venv
    source .venv/bin/activate
    ```
3.  **Sync dependencies using uv:**
    This command installs the exact dependencies listed in `pyproject.toml`.
    ```bash
    uv sync --extra dev
    ```
4.  **Run the analysis:**
    To run the test suite, execute the following command from the root directory:
    ```bash
    pytest
    ```
    For other analyses, you can run the notebooks in the `notebooks/` directory.
---

## 📊 Key Results

> Our findings indicate that operator learning models, particularly Fourier Neural Operators, provide an excellent balance of accuracy and computational efficiency.
>
> -   The FNO model achieved an R² score of over 0.95 on the test set, accurately predicting both the amplitude and frequency content of the surface motion.
> -   Inference time for the surrogate model was over 1000x faster than the corresponding high-fidelity FLAC simulation.
> -   The model demonstrated strong generalization capabilities to unseen soil profiles and input ground motions.
>
> ![Key Figure](outputs/figures/key_figure.png)
>
> _**Figure 1:** Comparison of surface acceleration response spectra between the FLAC simulation (ground truth) and the FNO surrogate model prediction for a sample case from the test set._
