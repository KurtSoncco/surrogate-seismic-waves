#!/bin/bash
# Speed/accuracy profiling for GIFNO-FDO-XT-LOGLO-POD on NCSA Delta (no wandb).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD
#   source ../GIFNO/delta_env.sh
#   sbatch delta_profile.sh --limit 500 --batches 1,2,4,8,16,32

#SBATCH --job-name=loglo_pod_prof
#SBATCH --account=bgpu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=00:40:00
#SBATCH --output=loglo_pod_prof.o%j
#SBATCH --error=loglo_pod_prof.e%j

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

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_H5_DIR="${DATA_ROOT}/h5"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"
# Isolated dir so profiling never clobbers sweep/train artifacts.
export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-${GIFNO_TF_DIR}/models/fdo_xt_loglo_pod/_profile}"

export GIFNO_LATENT_CHANNELS="${GIFNO_LATENT_CHANNELS:-128}"
export GIFNO_POD_NUM_MODES="${GIFNO_POD_NUM_MODES:-32}"

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

mkdir -p "${GIFNO_MODEL_DIR}"

load_delta_python() {
    if module is-loaded python 2>/dev/null; then
        return 0
    fi
    module load gcc python 2>/dev/null || module load python
}
load_delta_python

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== GIFNO-FDO-XT-LOGLO-POD profiling on Delta (no wandb) ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "GIFNO_MODEL_DIR=${GIFNO_MODEL_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Model: LATENT=${GIFNO_LATENT_CHANNELS} POD_MODES=${GIFNO_POD_NUM_MODES}"
echo "Args:  $*"
echo "=============================================="

python -u "${XT_DIR}/profile_speed.py" "$@"

echo "Done."
