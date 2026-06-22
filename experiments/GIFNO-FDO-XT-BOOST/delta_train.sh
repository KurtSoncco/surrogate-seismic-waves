#!/bin/bash
# GIFNO-FDO-XT-BOOST training on NCSA Delta.
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-BOOST
#   source ../GIFNO/delta_env.sh
#   sbatch delta_train.sh
#   bash delta_sweep.sh --variants sweep_variants_boost.tsv --limit 2000

#SBATCH --job-name=boost_train
#SBATCH --account=bgpu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=08:00:00
#SBATCH --output=boost_train.o%j
#SBATCH --error=boost_train.e%j

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

GIFNO_DIR="$(cd "${SCRIPT_DIR}/../GIFNO" && pwd)"
# shellcheck source=../GIFNO/delta_env.sh
[[ -f "${GIFNO_DIR}/delta_env.sh" ]] && source "${GIFNO_DIR}/delta_env.sh"
# shellcheck source=../GIFNO/delta_paths.sh
source "${GIFNO_DIR}/delta_paths.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v module &>/dev/null; then
    module reset 2>/dev/null || true
fi

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(resolve_gifno_data_root)"
XT_DIR="${SCRIPT_DIR}"

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

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_H5_DIR="${DATA_ROOT}/h5"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-${GIFNO_TF_DIR}/models/fdo_xt_boost}"
export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-${GIFNO_TF_DIR}/results/fdo_xt_boost}"

export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-16}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-4}"
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-false}"

if [[ -z "${GIFNO_PRETRAIN_CHECKPOINT:-}" ]]; then
    echo "ERROR: set GIFNO_PRETRAIN_CHECKPOINT to xt_lat128_d128/best_model.pt" >&2
    exit 1
fi
export GIFNO_PRETRAIN_CHECKPOINT="${GIFNO_PRETRAIN_CHECKPOINT}"

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

load_delta_python() {
    if module is-loaded python 2>/dev/null; then
        return 0
    fi
    module load gcc python 2>/dev/null || module load python
}
load_delta_python

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== GIFNO-FDO-XT-BOOST training on Delta ==="
echo "WANDB_PROJECT=gifno_fdo_xt_boost"
echo "GIFNO_PRETRAIN_CHECKPOINT=${GIFNO_PRETRAIN_CHECKPOINT}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "GIFNO_MODEL_DIR=${GIFNO_MODEL_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Speed:  BATCH_SIZE=${GIFNO_BATCH_SIZE} NUM_WORKERS=${GIFNO_NUM_WORKERS}"
echo "Args:     $*"
echo "===================================="

python -u "${XT_DIR}/main.py" "$@"

echo "Done. Model: ${GIFNO_MODEL_DIR}/best_model.pt"
echo "Results:    ${GIFNO_RESULTS_DIR}/"
