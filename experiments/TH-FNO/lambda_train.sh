#!/usr/bin/env bash
# TH-FNO direct-|TF| training on Lambda Labs (or local with GIFNO_DATA_ROOT).
#
#   # on Lambda after syncing this repo + gifno_data:
#   export GIFNO_DATA_ROOT=~/gifno_data
#   tmux new -s thfno
#   bash experiments/TH-FNO/lambda_train.sh --limit 2000 --epochs 40
#
# W&B: place experiments/GIFNO/lambda_secrets.env with WANDB_API_KEY=
# Project: th_fno   Run name: th_fno_direct_n2000 (override with THFNO_WANDB_RUN_NAME)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
DATA_ROOT="${GIFNO_DATA_ROOT:-${HOME}/gifno_data}"
THFNO_DIR="${PROJECT_ROOT}/experiments/TH-FNO"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"

SECRETS_FILE=""
for cand in \
    "${THFNO_DIR}/lambda_secrets.env" \
    "${GIFNO_DIR}/lambda_secrets.env"; do
    if [[ -f "${cand}" ]]; then
        SECRETS_FILE="${cand}"
        break
    fi
done
if [[ -n "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    set +a
    echo "Loaded secrets from ${SECRETS_FILE}"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_API_KEY//$'\r'/}"
    WANDB_API_KEY="${WANDB_API_KEY#"${WANDB_API_KEY%%[![:space:]]*}"}"
    WANDB_API_KEY="${WANDB_API_KEY%"${WANDB_API_KEY##*[![:space:]]}"}"
    export WANDB_API_KEY
fi

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_H5_DIR="${DATA_ROOT}/h5"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"
export THFNO_MODEL_DIR="${THFNO_DIR}/checkpoints/th_fno"
export THFNO_RESULTS_DIR="${THFNO_DIR}/results"
# D2 FAIL → direct |TF| (AGENTS §2 / §4)
export THFNO_PREDICT_MODE="${THFNO_PREDICT_MODE:-direct}"
export THFNO_AMPLITUDE_DOMAIN="${THFNO_AMPLITUDE_DOMAIN:-linear}"
export THFNO_RESIDUAL_MODE="${THFNO_RESIDUAL_MODE:-log_mult}"
export THFNO_TREND_FREQ_SCALE="${THFNO_TREND_FREQ_SCALE:-1.0}"
export THFNO_WANDB_RUN_NAME="${THFNO_WANDB_RUN_NAME:-th_fno_direct_n2000}"
export WANDB_PROJECT="${WANDB_PROJECT:-th_fno}"

# Allow --predict-mode on the CLI; also sync env if caller set THFNO_PREDICT_MODE only
MODE_FLAG=()
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --predict-mode)
            export THFNO_PREDICT_MODE="$2"
            MODE_FLAG=(--predict-mode "$2")
            shift 2
            ;;
        --predict-mode=*)
            export THFNO_PREDICT_MODE="${1#*=}"
            MODE_FLAG=(--predict-mode "${THFNO_PREDICT_MODE}")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
if [[ ${#MODE_FLAG[@]} -eq 0 ]]; then
    MODE_FLAG=(--predict-mode "${THFNO_PREDICT_MODE}")
fi
# Distinct default run name when residual
if [[ "${THFNO_PREDICT_MODE}" == "residual" && "${THFNO_WANDB_RUN_NAME}" == "th_fno_direct_n2000" ]]; then
    export THFNO_WANDB_RUN_NAME="th_fno_residual_n2000"
fi

for req in \
    "${GIFNO_H5_DIR}" \
    "${GIFNO_TF_DIR}/tf_per_sample.npy" \
    "${GIFNO_TF_DIR}/manifest.csv"; do
    if [[ ! -e "${req}" ]]; then
        echo "ERROR: missing required path: ${req}" >&2
        exit 1
    fi
done

mkdir -p "${THFNO_MODEL_DIR}" "${THFNO_RESULTS_DIR}"
cd "${THFNO_DIR}"

echo "=== TH-FNO training ==="
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "PREDICT_MODE=${THFNO_PREDICT_MODE} RESIDUAL_MODE=${THFNO_RESIDUAL_MODE}"
echo "AMPLITUDE_DOMAIN=${THFNO_AMPLITUDE_DOMAIN} TREND_FREQ_SCALE=${THFNO_TREND_FREQ_SCALE}"
echo "LOG_DELTA_C=${THFNO_LOG_DELTA_C:-} LOSS_TERM_NORM=${THFNO_LOSS_TERM_NORM:-1} ZERO_INIT=${THFNO_ZERO_INIT_RESIDUAL:-1}"
echo "WANDB_PROJECT=${WANDB_PROJECT} RUN=${THFNO_WANDB_RUN_NAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Args: ${MODE_FLAG[*]} ${ARGS[*]:-}"

if command -v uv >/dev/null 2>&1; then
    uv run --project ../GIFNO-FDO-XT python -u main.py "${MODE_FLAG[@]}" "${ARGS[@]:-}"
else
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
    python -u main.py "${MODE_FLAG[@]}" "${ARGS[@]:-}"
fi

echo "Done. Mode=${THFNO_PREDICT_MODE} Model: ${THFNO_MODEL_DIR}/../th_fno_${THFNO_PREDICT_MODE}/best_model.pt"
