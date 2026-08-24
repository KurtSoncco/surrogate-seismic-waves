#!/usr/bin/env bash
# Train tier2_pod64 on the full 7680 GIFNO set (publication target).
#
# Usage (local GPU):
#   bash run_full_7680_train.sh
#   bash run_full_7680_train.sh --limit 500   # smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BOX_DEFAULT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"

export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-$BOX_DEFAULT}"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-$GIFNO_DATA_ROOT/h5}"
export GIFNO_TF_DIR="${GIFNO_TF_DIR:-$GIFNO_DATA_ROOT/transfer_function}"
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-$HOME/surrogate-seismic-waves/checkpoints/tier2_pod64_full7680}"
export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-$GIFNO_MODEL_DIR/results}"

# tier2_pod64 recipe
export GIFNO_LATENT_CHANNELS=128
export GIFNO_POD_NUM_MODES=64
export GIFNO_NUM_FNO_LAYERS=5
export GIFNO_DEEPONET_LATENT_DIM=128
export GIFNO_LOSS_RADIAL_WEIGHT=0.25
export GIFNO_LOGLO_PATCH_SIZE=16,20
export GIFNO_LOGLO_HF_NOISE_ALPHA=0.025
export GIFNO_BAND_CURRICULUM=true
export GIFNO_BAND_CURRICULUM_MODE=convergence
export GIFNO_LOSS_BAND_BALANCED_WEIGHT=0.5
export GIFNO_BAND_CURRICULUM_LR_RESTART=true
export GIFNO_EARLY_STOP_PATIENCE=140
export GIFNO_LR_SCHED_PATIENCE=40
export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-2}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-2}"
# Full 7680 cache needs ~8GB+ RAM for grids alone; disable on laptop-class hosts.
export GIFNO_CACHE_DATASET="${GIFNO_CACHE_DATASET:-false}"
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-gifno_fdo_xt_loglo_pod}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-tier2_pod64_full7680}"

mkdir -p "$GIFNO_MODEL_DIR" "$GIFNO_RESULTS_DIR"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Full-7680 tier2_pod64 training ==="
echo "MODEL_DIR=$GIFNO_MODEL_DIR"
echo "DATA_ROOT=$GIFNO_DATA_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Args: $*"

cd "$SCRIPT_DIR"
# Remove stale POD so it is rebuilt for the full train split + 64 modes
rm -f "$GIFNO_MODEL_DIR/pod_modes.npy" "$GIFNO_MODEL_DIR/pod_mean.npy"
python -u main.py "$@"
