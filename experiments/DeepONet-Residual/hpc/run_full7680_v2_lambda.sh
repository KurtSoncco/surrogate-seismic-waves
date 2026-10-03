#!/usr/bin/env bash
# DeepONet-Residual v2: per-recorder branch, full-1000-freq training, TF composite loss.
#
# Architecture change vs v1 (global ResUNet branch) — do not warm-start from v1 ckpt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${GIFNO_DATA_ROOT:-$HOME/gifno_data}"

export GIFNO_DATA_ROOT="$DATA_ROOT"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-$DATA_ROOT/h5}"
export GIFNO_TF_DIR="${GIFNO_TF_DIR:-$DATA_ROOT/transfer_function}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-deeponet_residual_r_nom}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-per_rec_col_full7680_v2}"

CACHE_TAG="${CACHE_TAG:-full7680_seed42}"
LOG="${LOG:-$HOME/deeponet_full7680_v2.log}"

cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== DeepONet-Residual v2 (per-rec + TF loss) ==="
echo "DATA_ROOT=$GIFNO_DATA_ROOT"
echo "CACHE_TAG=$CACHE_TAG"
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

RUN_NAME="${RUN_NAME:-single_per_rec_col_full_R_nom_${CACHE_TAG}}"

echo "[2/2] Training per-rec column encoder R_nom..."
python -u train.py \
  --cache-tag "$CACHE_TAG" \
  --target R_nom \
  --branch-mode single_per_rec \
  --trunk-set full \
  --field-encoder column \
  --epochs "${EPOCHS:-300}" \
  --patience "${PATIENCE:-60}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-1e-3}" \
  --n-freq-train "${N_FREQ_TRAIN:-1000}" \
  --loss-tf-weight "${LOSS_TF_WEIGHT:-1.0}" \
  --selection-metric "${SELECTION_METRIC:-rel_l2_TF}" \
  --run-name "$RUN_NAME" \
  2>&1 | tee "$LOG"

echo "Training done. Log: $LOG"
echo "Checkpoint: $SCRIPT_DIR/checkpoints/${RUN_NAME}.pt"
