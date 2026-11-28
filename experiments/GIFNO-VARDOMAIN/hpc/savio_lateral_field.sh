#!/bin/bash
# Dense 1 m acceptance on the coord_reseed stage-2 checkpoint. One A5000.
#
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_lateral_field.sh
#
#SBATCH --job-name=vd_lateral
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
CK="${CK:-/global/scratch/users/kurtwal98/vardomain/checkpoints/coord_reseed/stage2}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/precision}"
# Optional overlay. Build it from the HTC H5 with dense_labels.py (drop the pads,
# stride 1 m) after savio_dense_recorders.sh. The gap test does not need it.
DENSE="${DENSE_NPZ:-}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
mkdir -p "${OUT}" /global/scratch/users/kurtwal98/vardomain/logs

[[ -f "${CK}/best_model.pt" ]] || { echo "missing ${CK}/best_model.pt" >&2; exit 1; }

extra=()
if [[ -n "${DENSE}" ]]; then
  extra+=(--dense-npz "${DENSE}")
fi

echo "=== lateral field ${CK} ==="
srun python -u "${VD_DIR}/lateral_field.py" \
  --checkpoint "${CK}" --cache "${CACHE}" --out "${OUT}" "${extra[@]}"
