#!/bin/bash
# Three existing timing rows, surface recorders every 1 m. HTC, not GPU.
# Uses the seiskit copy that honors xi_soil_base. The H5 is kept.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/timing15
#   sbatch /global/scratch/users/kurtwal98/vardomain/repo/experiments/GIFNO-VARDOMAIN/hpc/savio_dense_recorders.sh
#
#SBATCH --job-name=vd_dense
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/dense_recorders/logs/%A_%a.out
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/dense_recorders/logs/%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/global/scratch/users/kurtwal98/vardomain/dense_recorders}"
REPO="${REPO:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/seiskit/.venv}"
# The home seiskit on Savio rejects xi_soil_base. This copy is the local package
# that accepts it. Do not point the job back at ~/seiskit.
export SEISKIT_ROOT="${SEISKIT_ROOT:-/global/scratch/users/kurtwal98/vardomain/timing15/seiskit}"
export CASE_DIR="${CASE_DIR:-/global/scratch/users/kurtwal98/vardomain/timing15/case7a}"
if [[ ! -f "${CASE_DIR}/run_sections.py" ]]; then
  echo "missing ${CASE_DIR}/run_sections.py" >&2
  exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export BASIN_SOLVER="${BASIN_SOLVER:-UmfPack}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
WHEEL_LIB="$("${VENV}/bin/python" - <<'PY'
from pathlib import Path
import sys
lib = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "openseespylinux" / "lib"
print(lib)
PY
)"
export LD_LIBRARY_PATH="${WHEEL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

MANIFEST="${MANIFEST:-/global/scratch/users/kurtwal98/vardomain/timing15/timing_widths.csv}"
if [[ ! -f "${MANIFEST}" ]]; then
  MANIFEST="${REPO}/experiments/GIFNO-VARDOMAIN/hpc/timing_widths.csv"
fi
mkdir -p "${ROOT}/logs"
cd "${REPO}/experiments/GIFNO-VARDOMAIN/hpc"
python -u savio_dense_recorders.py \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --out-dir "${ROOT}" \
  --hx-m 1
