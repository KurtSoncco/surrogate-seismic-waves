#!/usr/bin/env bash
# Full-7680 ResUNet DeepONet-Residual (R_nom) on Lambda A100.
#
# Pipeline: signed cache (Haskell TF1D + fields) → train → W&B offline sync.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${GIFNO_DATA_ROOT:-$HOME/gifno_data}"

export GIFNO_DATA_ROOT="$DATA_ROOT"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-$DATA_ROOT/h5}"
export GIFNO_TF_DIR="${GIFNO_TF_DIR:-$DATA_ROOT/transfer_function}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-deeponet_residual_r_nom}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-resunet_full7680_lambda}"

CACHE_TAG="${CACHE_TAG:-full7680_seed42}"
INIT_CKPT="${INIT_CKPT:-$SCRIPT_DIR/checkpoints/single_resunet_full_R_nom_n1000_seed42.pt}"
LOG="${LOG:-$HOME/deeponet_full7680.log}"

cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== DeepONet-Residual full7680 ==="
echo "DATA_ROOT=$GIFNO_DATA_ROOT"
echo "CACHE_TAG=$CACHE_TAG"
echo "INIT_CKPT=$INIT_CKPT"
echo "WANDB_RUN=$WANDB_RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$SCRIPT_DIR"

CACHE_DIR="$SCRIPT_DIR/cache/${CACHE_TAG}"
echo "[1/2] Signed residual cache..."
if [[ -f "$CACHE_DIR/fields_rec.npy" ]] && [[ "${REBUILD_CACHE:-0}" != "1" ]]; then
  echo "Using existing cache: $CACHE_DIR"
else
  python -u residual_signed.py --cache-tag "$CACHE_TAG" --force
fi

echo "[2/2] Training ResUNet R_nom..."
python -u train.py \
  --cache-tag "$CACHE_TAG" \
  --target R_nom \
  --branch-mode single \
  --trunk-set full \
  --field-encoder resunet \
  --epochs "${EPOCHS:-300}" \
  --patience "${PATIENCE:-60}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --lr "${LR:-1e-3}" \
  --run-name "single_resunet_full_R_nom_${CACHE_TAG}" \
  --init-checkpoint "$INIT_CKPT"

echo "Training done. Sync W&B with: bash ~/wandb_sync.sh"
echo "Log: $LOG"
