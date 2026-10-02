# Surrogate Modeling of Seismic Waves

[![Project Status](https://img.shields.io/badge/Project%20Status-Active-brightgreen?style=for-the-badge)](https://github.com/KurtSoncco/surrogate-seismic-waves)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/KurtSoncco/surrogate-seismic-waves/actions/workflows/ci.yml/badge.svg)](https://github.com/KurtSoncco/surrogate-seismic-waves)
[![uv](https://img.shields.io/badge/uv-%3E%3D0.1.0-blue?style=for-the-badge)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Github stars](https://img.shields.io/github/stars/KurtSoncco/surrogate-seismic-waves?style=social)](https://github.com/KurtSoncco/surrogate-seismic-waves/stargazers)

> This repository develops surrogate models for seismic wave propagation and site-response prediction. Two shipped operators map 2D OpenSees soil-variability domains to transfer functions: **LOGLO-POD** (direct TF, best in-family IID) and **residual GINO** (Haskell leftover, best OOD). The goal is accurate transfer-function prediction with orders-of-magnitude faster inference than full physics runs.

[Research Questions](#-research-questions--hypothesis) • [Repository Layout](#-repository-layout) • [Methodology](#️-methodology) • [Data](#-data) • [Experiments](#-experiments) • [Key Results](#-key-results) • [How to Reproduce](#-how-to-reproduce)

---

## 🎯 Research Questions / Hypothesis

- How effective are operator-learning models versus classical 1D/2D physics simulators for predicting transfer functions?
- Can a 1D Haskell prior plus a residual operator improve OOD generalization on layered and dipping geometry?
- What is the accuracy vs. speed trade-off when generalizing to unseen soil profiles and frequency content?

---

## 📁 Repository Layout

```
surrogate-seismic-waves/
├── wave_surrogate/          # Core library: FNO, DAE, PCE, FLAC API, TTF utilities
├── experiments/
│   ├── GIFNO/               # Shared OpenSees loader, losses, Delta scripts
│   ├── GIFNO-FDO-XT-LOGLO-POD/  # LOGLO-POD full (best 2D IID model)
│   └── DeepONet-Residual/   # Residual GINO rebal FT (shipped leftover)
├── tests/                   # CI test suite (library + experiment tests)
└── pyproject.toml           # uv project config, Ruff, pytest
```

---

## 🛠️ Methodology

1. **Data generation**
   - **1D FLAC profiles:** Layered Vs/Vp/ρ profiles → transfer functions (pickle/parquet datasets).
   - **2D OpenSees runs (GIFNO):** H5 wavefield snapshots on a 500 m soil-variability strip with lateral recorders; transfer functions computed in preprocessing.

2. **Surrogate modeling**
   - **LOGLO-POD:** Dual-path spectral encoder + POD-DeepONet readout maps the 2D strip directly to recorder TFs.
   - **Residual GINO:** Geometry-aware Haskell nom plus a GNO+FNO leftover on signed `R = TF_2D − TF_1D`.
   - **Composite losses (GIFNO):** Masked relative Lp + optional H¹ frequency loss + band curriculum.

3. **Evaluation**
   - Regression: MSE, MAE, RMSE, R², relative L2.
   - Shape fidelity: per-sample and per-frequency Pearson correlation, H¹ frequency metrics.
   - Diagnostics: worst/median/best sample plots, spatial TF heatmaps, W&B logging.

---

## 💾 Data

Datasets are **not stored in this repository** due to size. Split sizes, tensor shapes, and which `n=2000` is which are in **[`DATA.md`](DATA.md)**.

| Source | Description | Used by |
|--------|-------------|---------|
| FLAC 1D profiles | Vs profiles + transfer functions (~1000 samples) | `wave_surrogate` |
| OpenSees H5 (Box / HPC) | 2D wavefield runs (`run_*.h5`) + derived TF cache | `GIFNO`, `GIFNO-FDO-XT-LOGLO-POD`, `DeepONet-Residual` |
| Local dummy data | Synthetic paths for unit tests (no Box mount) | `GIFNO/tests` |

**Local data access (GIFNO):** mount Box at `/mnt/box` (or `/mnt/box_lab`) or set:

```bash
export GIFNO_DATA_ROOT="/path/to/data"   # must contain h5/ and transfer_function/
```

On HPC (NCSA Delta, Savio), use the shell scripts in `experiments/GIFNO/` (`delta_run_all.sh`, `hpc/savio_train.sh`, `hpc/lambda_train.sh`).

---

## 🧪 Experiments

### `GIFNO-FDO-XT-LOGLO-POD` — LOGLO encoder + POD readout (2D, full)

Best **in-family IID** 2D OpenSees surrogate: **dual-path LOGLO spectral encoder** (depth-collapsed to 1D-along-x) + **POD-DeepONet readout**. Publication model: `tier2_pod64` trained on the full 7680-sample set (`checkpoints/tier2_pod64_full7680`, W&B `tier2_pod64_full7680`).

- **Input:** `(4, 128, 500)` — normalized Vs, zeta, x/z coords on the 500 m variability strip.
- **Output:** Transfer functions at 21 lateral recorders × 1000 frequencies.
- **Training:** Convergence band curriculum + composite loss (radial, H¹, band-balanced); W&B project `gifno_fdo_xt_loglo_pod`.
- **OOD checks:** `capability_check.py` vs seiskit `three_layer/` and `dipping/` experiments.

```bash
cd experiments/GIFNO-FDO-XT-LOGLO-POD
source ../GIFNO/delta_env.sh
uv run python capability_check.py --all          # OOD capability checks
bash run_full_7680_train.sh                      # full-dataset training
sbatch --time=24:00:00 delta_train.sh            # same on Delta
```

Shared infrastructure (data loader, metrics, Delta scripts) lives in `experiments/GIFNO/`.

### `DeepONet-Residual` — residual GINO rebal FT

Shipped leftover on geometry-aware Haskell nom: freeze-GNO fine-tune of mix **M7680** (`iid_frac=0.34`, three-layer val stop). Checkpoint: `experiments/DeepONet-Residual/checkpoints/M7680_gino_rebal_ft.pt`.

```bash
GIFNO_DATA_ROOT=data/gifno_screen \
GIFNO_OOD_DIPPING=data/gifno_screen/ood_dipping \
GIFNO_OOD_THREE_LAYER=data/gifno_screen/ood_three_layer \
uv run python experiments/DeepONet-Residual/scoring/eval_ood.py --split test
```

See [`experiments/DeepONet-Residual/README.md`](experiments/DeepONet-Residual/README.md).

### `GIFNO` — shared OpenSees pipeline

H5 data loading, TF preprocessing, metrics, and NCSA Delta deployment scripts used by LOGLO-POD.

```bash
cd experiments/GIFNO
uv run python main.py --limit 32    # shared-pipeline smoke test
```

### `wave_surrogate` — Core Package

Reusable implementations tested in CI:

- `models/fno/` — 1D FNO training pipeline
- `models/dae/` — Denoising autoencoder architectures
- `models/pce/` — Polynomial chaos expansions (JAX)
- `flac/` — FLAC API helpers
- `ttf/` — Transfer-function utilities (Kohmachi, acceleration → FAS)

---

## 📊 Key Results

### LOGLO-POD — 2D OpenSees operator (full)

On the 2000-sample screen hold-out (`tier2_pod64`, convergence band curriculum):

| Metric | Value |
|--------|-------|
| `test_rel_l2` | **0.302** |
| `test_pearson` | **0.919** |
| `test_pearson_mean` (per-recorder) | **0.939** |

That screen uses the first 2000 manifest rows, not the residual nested tests.

### Residual GINO vs LOGLO — nested leftover tests (shipped)

Same seed-42 slices (IID 150 / dipping 144 / three-layer 144):

| Model | IID rel L2 | dipping | three-layer |
|-------|-----------:|--------:|------------:|
| 1D Haskell nom | 0.565 | 0.601 | 0.730 |
| LOGLO-POD full7680 | **0.338** | 0.930 | 0.843 |
| GINO rebal FT | 0.353 | **0.322** | **0.525** |

LOGLO is the better in-family amplitude fit and still collapses on layered/dipping geometry. GINO (`experiments/DeepONet-Residual/checkpoints/M7680_gino_rebal_ft.pt`) is the leftover that holds OOD. See [`experiments/DeepONet-Residual/results/compare_gino_loglo/`](experiments/DeepONet-Residual/results/compare_gino_loglo/) and [`experiments/DeepONet-Residual/README.md`](experiments/DeepONet-Residual/README.md).

---

## 🚀 How to Reproduce

### 1. Clone and set up the environment

```bash
git clone https://github.com/KurtSoncco/surrogate-seismic-waves
cd surrogate-seismic-waves

pyenv local 3.11   # optional
uv venv
source .venv/bin/activate
uv sync --extra dev   # installs ruff, pytest, and all dependencies
```

### 2. Run CI checks locally (recommended before push)

```bash
uv run ruff check .
uv run pytest
```

CI runs the same steps on every push to `main` ([workflow](.github/workflows/ci.yml)).

### 3. Run an experiment

Point data paths via each experiment's `config.py` or environment variables, then:

```bash
# LOGLO-POD capability check or full training
cd experiments/GIFNO-FDO-XT-LOGLO-POD && uv run python capability_check.py --all

# Residual GINO OOD eval (default checkpoint = M7680_gino_rebal_ft)
uv run python experiments/DeepONet-Residual/scoring/eval_ood.py --split test
```

### 4. GIFNO on NCSA Delta (from WSL)

```bash
bash experiments/GIFNO/delta_run_all.sh
```

Requires Box mount (`go-lab`), Duo MFA for SSH, and W&B credentials on the cluster.

---

## 🔧 Development Notes

- **Package manager:** [uv](https://github.com/astral-sh/uv) with lockfile (`uv.lock`).
- **Linting:** Ruff (`uv run ruff check .`).
- **Tests:** pytest over `wave_surrogate`, `GIFNO`, `GIFNO-FDO-XT-LOGLO-POD`, and `DeepONet-Residual`.
- **Logging:** Weights & Biases for experiment tracking where configured.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
