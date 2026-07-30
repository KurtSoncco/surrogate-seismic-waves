#!/usr/bin/env bash
# Sequential A/B: direct |TF| vs log_mult residual on resonance-calibrated H_1D.
# Losses always on raw (linear) |TF| — not log-domain SmoothL1.
#
#   bash experiments/TH-FNO/lambda_ab_train.sh --limit 2000 --epochs 80
#
# W&B (project th_fno):
#   th_fno_direct_lin_cal_n2000
#   th_fno_residual_logmult_lin_cal_n2000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
DATA_ROOT="${GIFNO_DATA_ROOT:-${HOME}/gifno_data}"

export GIFNO_DATA_ROOT="${DATA_ROOT}"
# Raw |TF| losses (user requirement)
export THFNO_AMPLITUDE_DOMAIN=linear
export THFNO_RESIDUAL_MODE="${THFNO_RESIDUAL_MODE:-log_mult}"

# Calibrate resonance scale from GIFNO TF cache if not already set
SCALE_JSON="${SCRIPT_DIR}/results/diagnostics/trend_freq_scale.json"
if [[ -z "${THFNO_TREND_FREQ_SCALE:-}" ]]; then
    echo "=== Calibrating TREND_FREQ_SCALE from GIFNO ==="
    cd "${SCRIPT_DIR}"
    if command -v uv >/dev/null 2>&1; then
        uv run --project ../GIFNO-FDO-XT python -u diagnostics/calibrate_trend_freq_scale.py \
            --source gifno --limit "${THFNO_CALIB_LIMIT:-500}" --out "${SCALE_JSON}"
    else
        # shellcheck source=/dev/null
        source "${PROJECT_ROOT}/.venv/bin/activate"
        python -u diagnostics/calibrate_trend_freq_scale.py \
            --source gifno --limit "${THFNO_CALIB_LIMIT:-500}" --out "${SCALE_JSON}"
    fi
    export THFNO_TREND_FREQ_SCALE
    THFNO_TREND_FREQ_SCALE="$(python -c "import json; print(json.load(open('${SCALE_JSON}'))['TREND_FREQ_SCALE'])")"
    export THFNO_TREND_FREQ_SCALE
fi
echo "THFNO_TREND_FREQ_SCALE=${THFNO_TREND_FREQ_SCALE}"
echo "THFNO_AMPLITUDE_DOMAIN=${THFNO_AMPLITUDE_DOMAIN}"
echo "THFNO_RESIDUAL_MODE=${THFNO_RESIDUAL_MODE}"

echo "=== A/B 1/2: PREDICT_MODE=direct (linear loss, cal trend unused in forward) ==="
export THFNO_PREDICT_MODE=direct
export THFNO_WANDB_RUN_NAME="${THFNO_WANDB_RUN_NAME_DIRECT:-th_fno_direct_lin_cal_n2000}"
bash "${SCRIPT_DIR}/lambda_train.sh" --predict-mode direct "$@"

echo "=== A/B 2/2: PREDICT_MODE=residual log_mult on calibrated H_1D (linear loss) ==="
export THFNO_PREDICT_MODE=residual
export THFNO_WANDB_RUN_NAME="${THFNO_WANDB_RUN_NAME_RESIDUAL:-th_fno_residual_logmult_lin_cal_n2000}"
bash "${SCRIPT_DIR}/lambda_train.sh" --predict-mode residual "$@"

echo "=== A/B done ==="
echo "Compare W&B project th_fno:"
echo "  th_fno_direct_lin_cal_n2000  vs  th_fno_residual_logmult_lin_cal_n2000"
echo "Metrics JSON:"
echo "  ${SCRIPT_DIR}/results/gifno_test_metrics_direct.json"
echo "  ${SCRIPT_DIR}/results/gifno_test_metrics_residual.json"
