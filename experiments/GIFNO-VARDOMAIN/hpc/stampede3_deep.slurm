#!/bin/bash
# Optional 250 m sections on Stampede3. Writes to $SCRATCH/vardomain/deep250_pad,
# never to case7_sections/round0. Each run is 3000 x 250 m (2000 m interior +
# 500 m free-field pads). 28 cores/task so four 750k-element UmfPack jobs share
# a 128 GB node. Depth comes from the CSV ``depth_m`` column.
#
#   bash experiments/GIFNO-VARDOMAIN/hpc/submit_deep.sh
#
#SBATCH -J vd_pad
#SBATCH -o vd_pad_%j.o
#SBATCH -e vd_pad_%j.e
#SBATCH -p spr
#SBATCH -N 8
#SBATCH -n 896
#SBATCH -t 24:00:00
#SBATCH -A ECS24003

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VD_DIR="$(cd "${HPC_DIR}/.." && pwd)"
CASE_DIR="$(cd "${VD_DIR}/../case7a" && pwd)"
CASE_HPC="${CASE_DIR}/hpc"
PROJECT_ROOT="$(cd "${VD_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
export SEISKIT_ROOT="${SEISKIT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
MANIFEST="$(realpath "${MANIFEST:?set MANIFEST to deep250.csv}")"
OUT_DIR="${OUT_DIR:-${SCRATCH}/vardomain/deep250_pad}"
CORES_PER_TASK="${CORES_PER_TASK:-28}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 BASIN_SOLVER="${BASIN_SOLVER:-UmfPack}"

module load python3 pylauncher 2>/dev/null || module load pylauncher
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
mkdir -p "${OUT_DIR}"

if [[ "${TIMING:-0}" == "1" ]]; then
    /usr/bin/time -v python -u "${CASE_DIR}/run_sections.py" --manifest "${MANIFEST}" --index 0 \
        --timing "${TIMING_SECONDS:-1.0}" --out-dir "${OUT_DIR}/timing"
    exit 0
fi

TASKS="${OUT_DIR}/tasks_${SLURM_JOB_ID:-local}.txt"
python "${CASE_HPC}/make_section_tasks.py" --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" > "${TASKS}"
echo "$(date -Is) tasks=$(wc -l < "${TASKS}") nodes=${SLURM_NNODES:-1} cores/task=${CORES_PER_TASK} out=${OUT_DIR}"
[[ -s "${TASKS}" ]] || { echo "nothing to run"; exit 0; }

cd "${OUT_DIR}"
python - "${TASKS}" "${CORES_PER_TASK}" <<'PY'
import sys
import pylauncher

pylauncher.ClassicLauncher(sys.argv[1], cores=int(sys.argv[2]), debug="job")
PY
echo "$(date -Is) done; failures: $(cat "${OUT_DIR}/failures.log" 2>/dev/null | wc -l)"
