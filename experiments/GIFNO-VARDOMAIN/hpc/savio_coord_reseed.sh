#!/bin/bash
# From-scratch curriculum with float32 queries, a sub-station offset, whitened
# POD coefficients, and the lateral cutoff named by the periodogram. One A5000,
# 72 hours. Does not read early_width/stage2_rise50 as an init.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/repo
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_coord_reseed.sh
#
#SBATCH --job-name=vd_coord
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
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/coord_reseed}"
JSON="${LATERAL_JSON:-/global/scratch/users/kurtwal98/vardomain/precision/lateral_periodogram.json}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
mkdir -p "${OUT}" /global/scratch/users/kurtwal98/vardomain/logs

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
[[ -f "${JSON}" ]] || { echo "missing ${JSON}; run savio_coord_diag.sh first" >&2; exit 1; }

read -r HFP_KERNEL HFP_STRIDE LOCAL_MIN < <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["hfp_kernel"]), int(d["hfp_stride"]), float(d["local_min_wavelength"]))' "${JSON}")

echo "=== coord reseed -> ${OUT} ==="
echo "carrier cutoff hfp ${HFP_KERNEL}/${HFP_STRIDE} local_min_wavelength ${LOCAL_MIN}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/curriculum.py" \
  --cache "${CACHE}" --out "${OUT}" --resume \
  --jitter --whiten --curvature 0.001 \
  --hfp-kernel "${HFP_KERNEL}" --hfp-stride "${HFP_STRIDE}" \
  --local-min-wavelength "${LOCAL_MIN}"
