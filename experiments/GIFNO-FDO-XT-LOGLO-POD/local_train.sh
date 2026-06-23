#!/usr/bin/env bash
# Local smoke / dev training for GIFNO-FDO-XT-LOGLO-POD.
#
#   cd experiments/GIFNO-FDO-XT-LOGLO-POD
#   bash local_train.sh --limit 500
#
# Defaults to box_lab data when mounted; set GIFNO_DATA_ROOT to override.
# W&B: sources ../GIFNO/lambda_secrets.env when present (WANDB_API_KEY).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GIFNO_DIR="$(cd "${SCRIPT_DIR}/../GIFNO" && pwd)"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"

BOX_DATA="/mnt/box_lab/Projects/Neural Operator/data"
if [[ -z "${GIFNO_DATA_ROOT:-}" && -d "${BOX_DATA}/transfer_function" ]]; then
    export GIFNO_DATA_ROOT="${BOX_DATA}"
fi
if [[ -n "${GIFNO_DATA_ROOT:-}" ]]; then
    export GIFNO_H5_DIR="${GIFNO_H5_DIR:-${GIFNO_DATA_ROOT}/h5}"
    export GIFNO_TF_DIR="${GIFNO_TF_DIR:-${GIFNO_DATA_ROOT}/transfer_function}"
fi

SECRETS_FILE="${GIFNO_DIR}/lambda_secrets.env"
if [[ -f "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    set +a
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_API_KEY//$'\r'/}"
    WANDB_API_KEY="${WANDB_API_KEY#"${WANDB_API_KEY%%[![:space:]]*}"}"
    WANDB_API_KEY="${WANDB_API_KEY%"${WANDB_API_KEY##*[![:space:]]}"}"
    export WANDB_API_KEY
fi

export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-8}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-2}"
export GIFNO_LATENT_CHANNELS="${GIFNO_LATENT_CHANNELS:-128}"
export GIFNO_POD_NUM_MODES="${GIFNO_POD_NUM_MODES:-32}"
export GIFNO_LOSS_RADIAL_WEIGHT="${GIFNO_LOSS_RADIAL_WEIGHT:-0.25}"
export GIFNO_WANDB_RUN_NAME="${GIFNO_WANDB_RUN_NAME:-loglo_pod_smoke_500}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "${SCRIPT_DIR}"
exec "${VENV_PY}" main.py "$@"
