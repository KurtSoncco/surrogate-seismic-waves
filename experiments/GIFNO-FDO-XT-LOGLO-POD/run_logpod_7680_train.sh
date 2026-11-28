#!/usr/bin/env bash
# Train tier2_pod64 on log(|TF|) with an exp readout.
#
# The POD basis is log-space. GIFNO_LOG_TF_LOSS scores relative error there.
# Weights are not started by the basis build; run this on a GPU.
#
# Usage:
#   bash run_logpod_7680_train.sh
#   bash run_logpod_7680_train.sh --limit 500
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BOX_DEFAULT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"

export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-$BOX_DEFAULT}"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-$GIFNO_DATA_ROOT/h5}"
export GIFNO_TF_DIR="${GIFNO_TF_DIR:-$GIFNO_DATA_ROOT/transfer_function}"
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-$PROJECT_ROOT/checkpoints/logpod_full7680}"
export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-$GIFNO_MODEL_DIR/results}"

export GIFNO_POD_SPACE=log
export GIFNO_LOG_TF_LOSS=true
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
export GIFNO_CACHE_DATASET="${GIFNO_CACHE_DATASET:-false}"
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-gifno_fdo_xt_loglo_pod}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-logpod_full7680}"

mkdir -p "$GIFNO_MODEL_DIR" "$GIFNO_RESULTS_DIR"
if [[ -f "$GIFNO_MODEL_DIR/pod_space.txt" ]]; then
  space="$(tr -d '[:space:]' < "$GIFNO_MODEL_DIR/pod_space.txt")"
  if [[ "$space" != "log" ]]; then
    echo "refusing $GIFNO_MODEL_DIR: pod_space.txt is $space" >&2
    exit 1
  fi
fi
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== log |TF| POD training ==="
echo "MODEL_DIR=$GIFNO_MODEL_DIR"
echo "DATA_ROOT=$GIFNO_DATA_ROOT"
echo "POD_SPACE=$GIFNO_POD_SPACE  LOG_TF_LOSS=$GIFNO_LOG_TF_LOSS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Args: $*"

cd "$SCRIPT_DIR"
python -u main.py "$@"
