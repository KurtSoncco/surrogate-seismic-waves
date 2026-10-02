#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-$HOME/gifno_data}"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-$GIFNO_DATA_ROOT/h5}"
export GIFNO_TF_DIR="${GIFNO_TF_DIR:-$GIFNO_DATA_ROOT/transfer_function}"
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-$HOME/surrogate-seismic-waves/checkpoints/tier2_pod64_full7680}"
export GIFNO_POD_NUM_MODES=64 GIFNO_LATENT_CHANNELS=128 GIFNO_NUM_FNO_LAYERS=5 GIFNO_DEEPONET_LATENT_DIM=128
export SEISKIT_ROOT="${SEISKIT_ROOT:-$HOME/seiskit}"
export OOD_H5_ROOT="${OOD_H5_ROOT:-$HOME/surrogate-seismic-waves/data/ood_h5}"
export OOD_GT_ROOT="${OOD_GT_ROOT:-$HOME/surrogate-seismic-waves/checkpoints/ood_scores_full7680}"
cd "$PROJECT_ROOT" && source .venv/bin/activate
cd "$SCRIPT_DIR"
echo "=== OOD TF compare LOGLO vs DeepONet ==="
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null || true
python -u scoring/compare_tf_ood_loglo_vs_deeponet.py
