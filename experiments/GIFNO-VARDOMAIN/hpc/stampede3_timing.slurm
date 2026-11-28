#!/bin/bash
# Short OpenSees timings, one mesh per interior width, to replace the
# core-hour table. Each task is --timing seconds of motion on the full mesh,
# not a production record. Writes under $SCRATCH/vardomain/timing.
#
#   cd experiments/GIFNO-VARDOMAIN/hpc
#   MANIFEST=$PWD/timing_widths.csv sbatch stampede3_timing.slurm
#
#SBATCH -J vd_time
#SBATCH -o vd_time_%j.o
#SBATCH -e vd_time_%j.e
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 02:00:00
#SBATCH -A TG-CIV260015

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VD_DIR="$(cd "${HPC_DIR}/.." && pwd)"
CASE_DIR="$(cd "${VD_DIR}/../case7a" && pwd)"
PROJECT_ROOT="$(cd "${VD_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-/work2/09739/kurtsoncco1406/stampede3/seiskit/.venv}"
export SEISKIT_ROOT="${SEISKIT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
MANIFEST="$(realpath "${MANIFEST:?set MANIFEST to timing_widths.csv}")"
OUT_DIR="${OUT_DIR:-${SCRATCH}/vardomain/timing}"
CORES_PER_TASK="${CORES_PER_TASK:-28}"
TIMING_SECONDS="${TIMING_SECONDS:-0.5}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 BASIN_SOLVER="${BASIN_SOLVER:-UmfPack}"

module load python3 pylauncher 2>/dev/null || module load pylauncher
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
mkdir -p "${OUT_DIR}/logs"

python - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(open("${MANIFEST}")))
seen = {}
for i, row in enumerate(rows):
    seen.setdefault(str(int(float(row["lx_m"]))), i)
Path("${OUT_DIR}/timing_index.txt").write_text("\n".join(f"{w} {i}" for w, i in seen.items()) + "\n")
print("timing indices", seen)
PY

TASKS="${OUT_DIR}/tasks_${SLURM_JOB_ID:-local}.txt"
: > "${TASKS}"
while read -r width index; do
  echo "${VENV_PATH}/bin/python -u ${CASE_DIR}/run_sections.py --manifest ${MANIFEST} --index ${index} --timing ${TIMING_SECONDS} --out-dir ${OUT_DIR} > ${OUT_DIR}/logs/w${width}.log 2>&1" >> "${TASKS}"
done < "${OUT_DIR}/timing_index.txt"
echo "$(date -Is) tasks=$(wc -l < "${TASKS}") cores/task=${CORES_PER_TASK}"

cd "${OUT_DIR}"
python - "${TASKS}" "${CORES_PER_TASK}" <<'PY'
import sys
import pylauncher

pylauncher.ClassicLauncher(sys.argv[1], cores=int(sys.argv[2]), debug="job")
PY
echo "$(date -Is) timing done"
