#!/bin/bash
# 4 m periodogram on the shipped stage-2 weights. One A5000, two hours.
# Reads the checkpoint and does not update it. The reseed job waits on the JSON.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/repo
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_coord_diag.sh
#
#SBATCH --job-name=vd_coord_diag
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
CK="${CK:-/global/scratch/users/kurtwal98/vardomain/checkpoints/early_width/stage2_rise50}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/precision}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
mkdir -p "${OUT}" /global/scratch/users/kurtwal98/vardomain/logs

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
[[ -f "${CK}/best_model.pt" ]] || { echo "missing ${CK}/best_model.pt" >&2; exit 1; }

echo "=== lateral periodogram ${CK} -> ${OUT} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/precision_query_shift.py" --periodogram \
  --checkpoint "${CK}" --cache "${CACHE}" --out "${OUT}"
