#!/bin/bash
# Stage 0 with the depth code and the spectral basis locked by the probes.
# Input is a frozen PCA-16 of the soil column. Targets are 128-mode
# sqrt(f / 1 Hz) POD codes of the residual. The loss is sqrt(f)-weighted
# log-L1 on the reconstructed residual. Validation logs the low, mid, and
# high bands next to the POD reconstruction floor.
#
# Writes to scratch checkpoints/pca16_stage0. Does not touch coord_reseed
# or the shipped checkpoints.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/repo
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_pca16_stage0.sh
#
#SBATCH --job-name=vd_pca16
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
CK_OUT="${CK_OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/pca16_stage0}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || {
  echo "missing ${CACHE}/freq.npy" >&2
  exit 1
}

echo "=== PCA-16 residual stage 0 -> ${CK_OUT} ==="
echo "CACHE=${CACHE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" \
  --out "${CK_OUT}" \
  --stage 0 \
  --target residual \
  --seed 0 \
  --readout pod \
  --depth-pca \
  --encoding metric \
  --jitter \
  --whiten \
  --local-min-wavelength 10.0
