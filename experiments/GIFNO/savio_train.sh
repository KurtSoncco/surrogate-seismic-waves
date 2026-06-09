#!/bin/bash
#SBATCH --job-name=gifno_train
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=24:00:00
#SBATCH --output=gifno_train.o%j
#SBATCH --error=gifno_train.e%j

set -euo pipefail

# --- Paths (Savio) ---
PROJECT_ROOT="${PROJECT_ROOT:-/global/home/users/kurtwal98/surrogate-seismic-waves}"
SCRATCH_DATA="${SCRATCH_DATA:-/global/scratch/users/kurtwal98/neural_operator_data}"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"

# Expected layout under SCRATCH_DATA:
#   h5/run_*.h5
#   transfer_function/tf_per_sample.npy
#   transfer_function/freq.npy
#   transfer_function/manifest.csv
#   transfer_function/recorder_x_idx.npy
#   transfer_function/models/   (written during training)
#   transfer_function/results/  (written during evaluation)

export GIFNO_DATA_ROOT="${SCRATCH_DATA}"
export GIFNO_H5_DIR="${SCRATCH_DATA}/h5"
export GIFNO_TF_DIR="${SCRATCH_DATA}/transfer_function"
export GIFNO_MODEL_DIR="${GIFNO_TF_DIR}/models"
export GIFNO_RESULTS_DIR="${GIFNO_TF_DIR}/results"

# --- Sanity checks ---
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

# --- Modules ---
module purge
module load python/3.11
module load cuda/12.4

# --- Run ---
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== GIFNO training on Savio ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "TF cache: ${GIFNO_TF_DIR}"
echo "H5 dir:   ${GIFNO_H5_DIR} ($(ls "${GIFNO_H5_DIR}"/run_*.h5 2>/dev/null | wc -l) files)"
echo "==============================="

srun python -u "${GIFNO_DIR}/main.py"

echo "Done. Model: ${GIFNO_MODEL_DIR}/best_model.pt"
echo "Results:    ${GIFNO_RESULTS_DIR}/"
