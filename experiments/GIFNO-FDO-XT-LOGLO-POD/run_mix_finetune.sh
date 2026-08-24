#!/usr/bin/env bash
# Mix-finetune from a full-data (or n2000) checkpoint on GIFNO + OOD train split.
#
# Prerequisites:
#   1. score_ood_campaign.py has cached GT TFs (or prepare will compute them)
#   2. prepare_ood_mix_data.py has written mix_manifest / mix_tf
#
# Usage:
#   bash run_mix_finetune.sh
#   GIFNO_INIT_CHECKPOINT=.../best_model.pt bash run_mix_finetune.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BOX_DEFAULT="/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data"
MIX_DATA="${MIX_DATA:-$HOME/surrogate-seismic-waves/checkpoints/mix_finetune_data}"
INIT_CKPT="${GIFNO_INIT_CHECKPOINT:-$HOME/surrogate-seismic-waves/checkpoints/tier2_pod64_full7680/best_model.pt}"
# Fall back to n2000 if full not ready
if [[ ! -f "$INIT_CKPT" ]]; then
  INIT_CKPT="$HOME/surrogate-seismic-waves/checkpoints/tier2_pod64_n2000/best_model.pt"
fi

export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-$BOX_DEFAULT}"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-$GIFNO_DATA_ROOT/h5}"
# Point TF cache at the mix dataset
export GIFNO_TF_DIR="$MIX_DATA"
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-$HOME/surrogate-seismic-waves/checkpoints/mix_finetune_ood}"
export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-$GIFNO_MODEL_DIR/results}"

export GIFNO_LATENT_CHANNELS=128
export GIFNO_POD_NUM_MODES=64
export GIFNO_NUM_FNO_LAYERS=5
export GIFNO_DEEPONET_LATENT_DIM=128
export GIFNO_LOSS_RADIAL_WEIGHT=0.25
export GIFNO_BAND_CURRICULUM=true
export GIFNO_BAND_CURRICULUM_MODE=convergence
export GIFNO_LOSS_BAND_BALANCED_WEIGHT=0.5
export GIFNO_BAND_CURRICULUM_LR_RESTART=true
export GIFNO_EARLY_STOP_PATIENCE=80
export GIFNO_LR_SCHED_PATIENCE=30
export GIFNO_NUM_EPOCHS="${GIFNO_NUM_EPOCHS:-400}"
export GIFNO_INIT_CHECKPOINT="$INIT_CKPT"
export GIFNO_LEARNING_RATE="${GIFNO_LEARNING_RATE:-3e-4}"
export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-4}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-2}"
export GIFNO_CACHE_DATASET=true
export GIFNO_USE_AMP=true
export WANDB_PROJECT="${WANDB_PROJECT:-gifno_fdo_xt_loglo_pod}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mix_finetune_ood}"

cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
cd "$SCRIPT_DIR"

if [[ ! -f "$MIX_DATA/mix_tf_per_sample.npy" ]]; then
  echo "[mix] Preparing mix dataset..."
  python -u prepare_ood_mix_data.py --out-dir "$MIX_DATA"
fi

# Symlink expected filenames if prepare wrote mix_* names
mkdir -p "$MIX_DATA"
if [[ ! -f "$MIX_DATA/tf_per_sample.npy" ]]; then
  ln -sfn mix_tf_per_sample.npy "$MIX_DATA/tf_per_sample.npy"
fi
if [[ ! -f "$MIX_DATA/manifest.csv" ]]; then
  ln -sfn mix_manifest.csv "$MIX_DATA/manifest.csv"
fi

mkdir -p "$GIFNO_MODEL_DIR" "$GIFNO_RESULTS_DIR"
# Rebuild POD on the mix train split
rm -f "$GIFNO_MODEL_DIR/pod_modes.npy" "$GIFNO_MODEL_DIR/pod_mean.npy"

echo "=== Mix-finetune ==="
echo "INIT_CKPT=$INIT_CKPT"
echo "MIX_DATA=$MIX_DATA"
echo "MODEL_DIR=$GIFNO_MODEL_DIR"

# Train from scratch on mix is safer than mismatched POD warm-start; if init
# exists we still train with fresh POD (branch head re-learns coefficients).
# Optional weight init for encoder only is left as a future hook.
python -u main.py "$@"

# Evaluate hold-out OOD with the new checkpoint
echo "=== Hold-out OOD eval ==="
export GIFNO_MODEL_DIR
python -u eval_ood_holdout.py --mix-data "$MIX_DATA" --checkpoint "$GIFNO_MODEL_DIR/best_model.pt"
