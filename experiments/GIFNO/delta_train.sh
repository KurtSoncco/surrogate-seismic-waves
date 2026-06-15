#!/bin/bash
# GIFNO training on NCSA Delta (gpuA100x4 / gpuA40x4).
#
# --- One-time setup (on login node) ---
#   git clone https://github.com/KurtSoncco/surrogate-seismic-waves.git
#   cd surrogate-seismic-waves
#   bash experiments/GIFNO/delta_setup.sh
#
# --- Smoke test (interactive GPU, ~30 min) ---
#   srun --account=$DELTA_ACCOUNT --partition=gpuA100x4-interactive \
#     --gpus-per-node=1 --cpus-per-task=4 --mem=32g --time=00:30:00 \
#     bash experiments/GIFNO/delta_train.sh --limit 50
#
# --- Full training (batch) ---
#   cd experiments/GIFNO && sbatch delta_train.sh
#
# --- Local data upload (from WSL, Box mounted) ---
#   bash experiments/GIFNO/delta_rsync_from_local.sh
#
# W&B: experiments/GIFNO/lambda_secrets.env (gitignored) — copy to Delta scratch.

#SBATCH --job-name=gifno_train
#SBATCH --account=bgpu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --output=gifno_train.o%j
#SBATCH --error=gifno_train.e%j

set -euo pipefail

# sbatch copies this script to /var/spool/slurmd/...; BASH_SOURCE points there, not the repo.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=delta_env.sh
[[ -f "${SCRIPT_DIR}/delta_env.sh" ]] && source "${SCRIPT_DIR}/delta_env.sh"
# shellcheck source=delta_paths.sh
source "${SCRIPT_DIR}/delta_paths.sh"

# In batch jobs, module reset sets $SCRATCH / $WORK for the allocation.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v module &>/dev/null; then
    module reset 2>/dev/null || true
fi

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(resolve_gifno_data_root)"
GIFNO_DIR="${SCRIPT_DIR}"

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

load_delta_python() {
    if module is-loaded python 2>/dev/null; then
        return 0
    fi
    module load gcc python 2>/dev/null || module load python
}
load_delta_python

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== GIFNO training on Delta ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "TF cache: ${GIFNO_TF_DIR}"
echo "H5 dir:   ${GIFNO_H5_DIR} ($(ls "${GIFNO_H5_DIR}"/run_*.h5 2>/dev/null | wc -l) files)"
echo "Args:     $*"
echo "==============================="

python -u "${GIFNO_DIR}/main.py" "$@"

echo "Done. Model: ${GIFNO_MODEL_DIR}/best_model.pt"
echo "Results:    ${GIFNO_RESULTS_DIR}/"
