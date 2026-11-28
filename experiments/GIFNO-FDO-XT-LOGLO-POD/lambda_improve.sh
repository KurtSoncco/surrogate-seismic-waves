#!/usr/bin/env bash
# E0 POD ceiling (CPU) then Bet A residual pretrain then Bet B outcrop+prior12.
# Assumes lambda_dual_head.sh setup already ran on this box.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
SCREEN="${PROJECT_ROOT}/data/gifno_screen"
CK="${PROJECT_ROOT}/checkpoints/case7_generalization"

activate_venv() { source "${PROJECT_ROOT}/.venv/bin/activate"; }

e0() {
  activate_venv
  echo "=== E0 outcrop mix POD ceiling ==="
  "$PY" "${SCRIPT_DIR}/pod_ceiling_mix.py" \
    --train "${SCREEN}/outcrop_tf/outcrop/mix_pretrain" \
    --holdout "${SCREEN}/outcrop_tf/outcrop/mix_pretrain/holdout" \
    --modes 32,64,96,128 \
    --out-dir "${CK}/pod_ceiling/outcrop"
  echo "=== E0 residual mix POD ceiling ==="
  "$PY" "${SCRIPT_DIR}/pod_ceiling_mix.py" \
    --train "${SCREEN}/outcrop_tf/residual/mix_pretrain" \
    --holdout "${SCREEN}/outcrop_tf/residual/mix_pretrain/holdout" \
    --modes 32,64,96,128 \
    --out-dir "${CK}/pod_ceiling/residual"
}

bet_a() {
  echo "=== Bet A residual STAGE=pretrain ==="
  TARGETS=residual STAGE=pretrain SEEDS="${SEEDS:-0}" \
    GIFNO_SELECTION_METRIC=pearson_trusted \
    bash "${SCRIPT_DIR}/lambda_dual_head.sh" run
}

bet_b() {
  echo "=== Bet B outcrop PRIOR_CHANNELS=12 STAGE=pretrain ==="
  activate_venv
  export IID_DIR="${IID_DIR:-${SCREEN}/transfer_function}"
  export DIP_DIR="${DIP_DIR:-${SCREEN}/ood_dipping}"
  export TL_DIR="${TL_DIR:-${SCREEN}/ood_three_layer}"
  export CORNER_DIR="${CORNER_DIR:-${SCREEN}/corner_is}"
  export SECTION_DIRS="${SECTION_DIRS:-${PROJECT_ROOT}/data/case7a/stampede_round0/windows_r0}"
  export TF_TARGET=outcrop
  export DATA_OUT="${SCREEN}/outcrop_tf/outcrop"
  export CK_OUT="${CK}/outcrop_prior12"
  export STAGE=pretrain
  export SEEDS="${SEEDS:-0}"
  export GIFNO_PRIOR_CHANNELS=12
  export GIFNO_SELECTION_METRIC=pearson_trusted
  export GIFNO_POD_LAYOUT=shared
  export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-64}"
  export GIFNO_USE_AMP=true
  export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-8}"
  export GIFNO_NUM_EPOCHS="${GIFNO_NUM_EPOCHS:-300}"
  export GIFNO_WINDOW_SHIFT_MAX=6
  export GIFNO_WINDOW_CONSISTENCY_WEIGHT=0.1
  export GIFNO_NZ_MAX=160
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTHON="$PY"
  mkdir -p "${CK_OUT}"
  cd "${SCRIPT_DIR}"
  bash "${SCRIPT_DIR}/run_mix_finetune.sh"
}

svlab() {
  activate_venv
  echo "=== SVLAB dense 5 m on residual pretrain_s0 ==="
  "$PY" "${PROJECT_ROOT}/experiments/case7a/score_svlab.py" --no-wipe \
    --out-dir "${CK}/residual/pretrain_s0/svlab"
}

cmd="${1:-all}"
case "$cmd" in
  e0) e0 ;;
  a) bet_a ;;
  b) bet_b; svlab ;;
  svlab) svlab ;;
  all) e0; bet_a; bet_b; svlab ;;
  gpu) bet_b; svlab ;;
  *) echo "usage: $0 {e0|a|b|svlab|gpu|all}" >&2; exit 1 ;;
esac
