#!/usr/bin/env bash
# Session N+1 §B1 — seeded A/B: direct+cal-trend vs log_mult+cal-trend.
# ONE variable: residual mode. Matched trend, matched linear loss, matched C1–C3.
#
#   bash experiments/TH-FNO/lambda_b1_seeded_ab.sh --limit 2000 --epochs 80
#
# Seeds default: 0 1 2 3 4 (override THFNO_B1_SEEDS="0 1 2")
# Writes per-seed JSON under results/b1_seeded/ and a gate summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
DATA_ROOT="${GIFNO_DATA_ROOT:-${HOME}/gifno_data}"
OUT_DIR="${SCRIPT_DIR}/results/b1_seeded"
SEEDS_STR="${THFNO_B1_SEEDS:-0 1 2 3 4}"
read -r -a SEEDS <<< "${SEEDS_STR}"

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export THFNO_AMPLITUDE_DOMAIN=linear
export THFNO_RESIDUAL_MODE=log_mult
export THFNO_LOSS_TERM_NORM="${THFNO_LOSS_TERM_NORM:-1}"
export THFNO_ZERO_INIT_RESIDUAL="${THFNO_ZERO_INIT_RESIDUAL:-1}"
export THFNO_LOG_DELTA_C="${THFNO_LOG_DELTA_C:-$(python -c 'import math; print(math.log(3.0))')}"
export WANDB_PROJECT="${WANDB_PROJECT:-th_fno}"

# Calibrate if unset
SCALE_JSON="${SCRIPT_DIR}/results/diagnostics/trend_freq_scale.json"
if [[ -z "${THFNO_TREND_FREQ_SCALE:-}" ]]; then
    if [[ -f "${SCALE_JSON}" ]]; then
        THFNO_TREND_FREQ_SCALE="$(python -c "import json; print(json.load(open('${SCALE_JSON}'))['TREND_FREQ_SCALE'])")"
    else
        THFNO_TREND_FREQ_SCALE=0.938
    fi
    export THFNO_TREND_FREQ_SCALE
fi
echo "=== B1 seeded A/B ==="
echo "SEEDS=${SEEDS[*]}"
echo "TREND_FREQ_SCALE=${THFNO_TREND_FREQ_SCALE}"
echo "LOG_DELTA_C=${THFNO_LOG_DELTA_C} LOSS_TERM_NORM=${THFNO_LOSS_TERM_NORM} ZERO_INIT=${THFNO_ZERO_INIT_RESIDUAL}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=("$@")

for seed in "${SEEDS[@]}"; do
    echo "=== seed ${seed}: direct ==="
    export THFNO_PREDICT_MODE=direct
    export THFNO_WANDB_RUN_NAME="th_fno_b1_direct_s${seed}"
    bash "${SCRIPT_DIR}/lambda_train.sh" --predict-mode direct --seed "${seed}" "${EXTRA_ARGS[@]}"
    # Move metrics into b1 folder
    src="${SCRIPT_DIR}/results/gifno_test_metrics_direct_s${seed}.json"
    if [[ -f "${src}" ]]; then
        cp "${src}" "${OUT_DIR}/direct_s${seed}.json"
    fi

    echo "=== seed ${seed}: residual log_mult ==="
    export THFNO_PREDICT_MODE=residual
    export THFNO_WANDB_RUN_NAME="th_fno_b1_residual_s${seed}"
    bash "${SCRIPT_DIR}/lambda_train.sh" --predict-mode residual --seed "${seed}" "${EXTRA_ARGS[@]}"
    src="${SCRIPT_DIR}/results/gifno_test_metrics_residual_s${seed}.json"
    if [[ -f "${src}" ]]; then
        cp "${src}" "${OUT_DIR}/residual_s${seed}.json"
    fi
done

echo "=== B1 gate summary ==="
cd "${SCRIPT_DIR}"
if command -v uv >/dev/null 2>&1; then
    uv run --project ../GIFNO-FDO-XT python -u diagnostics/summarize_b1_gate.py --dir "${OUT_DIR}"
else
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
    python -u diagnostics/summarize_b1_gate.py --dir "${OUT_DIR}"
fi

echo "=== B1 done ==="
