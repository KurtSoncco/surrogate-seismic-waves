#!/bin/bash
# Metric LOGLO curriculum on one Savio GPU. Same command as the Lambda A10 run.
# Writes only to checkpoints/vardomain. The cache must already be on scratch.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/repo
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_train.sh
#
#SBATCH --job-name=vd_loglo
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
CK_OUT="${CK_OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints}"
module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || {
  echo "missing ${CACHE}/freq.npy: rsync the cache or run build_cache.py on scratch" >&2
  exit 1
}

echo "=== vardomain Savio curriculum -> ${CK_OUT} ==="
echo "CACHE=${CACHE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/curriculum.py" --cache "${CACHE}" --out "${CK_OUT}"
