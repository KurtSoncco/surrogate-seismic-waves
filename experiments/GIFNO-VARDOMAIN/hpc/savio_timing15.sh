#!/bin/bash
# 15 s OpenSees timings at 500, 1000, 1500, and 2000 m.
# One array task per interior width. The H5 is deleted after wall_s is read.
# Column and surrogate times stay on the GPU machine; this job is the solver only.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/timing15
#   sbatch savio_timing15.sh
#
#SBATCH --job-name=vd_time15
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/timing15/logs/%A_%a.out
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/timing15/logs/%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/global/scratch/users/kurtwal98/vardomain/timing15}"
VENV="${VENV:-/global/home/users/kurtwal98/seiskit/.venv}"
# The home seiskit on Savio rejects xi_soil_base. This copy is the local package
# that accepts it. Do not point the job back at ~/seiskit.
export SEISKIT_ROOT="${SEISKIT_ROOT:-${ROOT}/seiskit}"
export CASE_DIR="${ROOT}/case7a"
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

cd "${ROOT}"
python -u savio_timing15.py \
  --manifest "${ROOT}/timing_widths.csv" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --out-dir "${ROOT}"
