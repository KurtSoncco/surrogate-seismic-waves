#!/usr/bin/env bash
# D4: RV-inclusive GIFNO-FDO-XT retrain (coverage control) on Lambda.
#
# Upsamples short-rH / high-aHV rows by duplicating them in an ephemeral
# training manifest overlay via env flags consumed by train_rv_inclusive.py.
#
#   bash experiments/TH-FNO/lambda_d4_gifno_rv_inclusive.sh --limit 500 --epochs 30

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
DATA_ROOT="${GIFNO_DATA_ROOT:-${HOME}/gifno_data}"
THFNO_DIR="${PROJECT_ROOT}/experiments/TH-FNO"
XT_DIR="${PROJECT_ROOT}/experiments/GIFNO-FDO-XT"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"

SECRETS_FILE="${GIFNO_DIR}/lambda_secrets.env"
if [[ -f "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    set +a
fi

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_H5_DIR="${DATA_ROOT}/h5"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"
export GIFNO_MODEL_DIR="${THFNO_DIR}/checkpoints/d4_gifno_rv_inclusive"
export GIFNO_RESULTS_DIR="${THFNO_DIR}/results/d4_gifno_rv_inclusive"
export GIFNO_LATENT_CHANNELS=128
export GIFNO_DEEPONET_LATENT_DIM=128

for req in \
    "${GIFNO_H5_DIR}" \
    "${GIFNO_TF_DIR}/tf_per_sample.npy" \
    "${GIFNO_TF_DIR}/manifest.csv"; do
    if [[ ! -e "${req}" ]]; then
        echo "ERROR: missing ${req}" >&2
        exit 1
    fi
done

mkdir -p "${GIFNO_MODEL_DIR}" "${GIFNO_RESULTS_DIR}"
cd "${THFNO_DIR}"

echo "=== D4 RV-inclusive GIFNO-XT retrain ==="
if command -v uv >/dev/null 2>&1; then
    uv run --project ../GIFNO-FDO-XT python -u diagnostics/d4_rv_inclusive_train.py "$@"
else
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
    python -u diagnostics/d4_rv_inclusive_train.py "$@"
fi
