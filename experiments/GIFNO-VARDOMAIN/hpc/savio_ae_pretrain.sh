#!/bin/bash
# Offline depth and spectral autoencoders. One A5000.
# Default is the one-hour subsample. The full cache is the same script:
#
#   sbatch --time=04:00:00 --job-name=vd_ae_full \
#     --export=ALL,LIMIT_CURVES=0,LIMIT_COLUMNS=0,OUT=/global/scratch/users/kurtwal98/vardomain/checkpoints/ae_pretrain/full \
#     experiments/GIFNO-VARDOMAIN/hpc/savio_ae_pretrain.sh
#
# Writes report.json with 0.1–10 Hz holdout error against POD and an
# unscored 0.001–50 Hz decoder query. Does not train the lateral stack.
#SBATCH --job-name=vd_ae_pre
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/ae_pretrain/subsample}"
LIMIT_CURVES="${LIMIT_CURVES:-2048}"
LIMIT_COLUMNS="${LIMIT_COLUMNS:-512}"
EPOCHS="${EPOCHS:-40}"

case "${OUT}" in
  *"/nonlinear"*|*"/early_width"*|*"/coord_reseed"*)
    echo "refusing to write into ${OUT}" >&2
    exit 1
    ;;
esac

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
mkdir -p "${OUT}"

echo "=== ae pretrain curves=${LIMIT_CURVES} columns=${LIMIT_COLUMNS} epochs=${EPOCHS} -> ${OUT} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/pretrain_ae.py" \
  --cache "${CACHE}" \
  --out "${OUT}" \
  --limit-curves "${LIMIT_CURVES}" \
  --limit-columns "${LIMIT_COLUMNS}" \
  --epochs "${EPOCHS}" \
  --ranks 64 32
