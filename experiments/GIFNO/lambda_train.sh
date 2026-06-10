#!/bin/bash
# Run on Lambda Labs GPU instance (e.g. gpu_1x_a10):
#
#   ssh ubuntu@<LAMBDA_IP>
#   cd ~/surrogate-seismic-waves && git pull
#   tmux new -s gifno
#   bash experiments/GIFNO/lambda_train.sh          # full training
#   bash experiments/GIFNO/lambda_train.sh --limit 10   # smoke test
#
# W&B: copy lambda_secrets.env.example → lambda_secrets.env (gitignored) with your API key.
#
# Detach tmux: Ctrl+B then D
# Reattach:    tmux attach -t gifno
# Logs:        tail -f ~/surrogate-seismic-waves/wandb/latest-run/files/output.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
DATA_ROOT="${GIFNO_DATA_ROOT:-${HOME}/gifno_data}"
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
export GIFNO_MODEL_DIR="${GIFNO_TF_DIR}/models"
export GIFNO_RESULTS_DIR="${GIFNO_TF_DIR}/results"

for req in \
    "${GIFNO_H5_DIR}" \
    "${GIFNO_TF_DIR}/tf_per_sample.npy" \
    "${GIFNO_TF_DIR}/manifest.csv" \
    "${GIFNO_TF_DIR}/recorder_x_idx.npy"; do
    if [[ ! -e "${req}" ]]; then
        echo "ERROR: missing required path: ${req}" >&2
        exit 1
    fi
done

mkdir -p "${GIFNO_MODEL_DIR}" "${GIFNO_RESULTS_DIR}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== GIFNO training on Lambda ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "TF cache: ${GIFNO_TF_DIR}"
echo "H5 dir:   ${GIFNO_H5_DIR} ($(ls "${GIFNO_H5_DIR}"/run_*.h5 2>/dev/null | wc -l) files)"
echo "Args:     $*"
echo "================================"

python -u "${GIFNO_DIR}/main.py" "$@"

echo "Done. Model: ${GIFNO_MODEL_DIR}/best_model.pt"
echo "Results:    ${GIFNO_RESULTS_DIR}/"
