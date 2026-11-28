#!/bin/bash
# Whole-section OpenSees runs on Stampede3 SPR nodes through pylauncher.
#
#   cd ~/surrogate-seismic-waves/experiments/case7a/hpc
#   MANIFEST=../sections/round0.csv sbatch -N 4 stampede3_sections.slurm
#   TIMING=1 MANIFEST=../sections/round0.csv sbatch -N 1 -t 02:00:00 stampede3_sections.slurm
#
# One 2000 m x 150 m section at 1 m cells is one single-threaded task:
# 300k quads, 2.8 GB peak RSS, 8.4 s/step on a WSL desktop core, so about
# 3.5 h for 1500 steps (15 s at 0.01 s). CORES_PER_TASK=3 packs 37 tasks
# (~104 GB) on an SPR node (112 cores, 128 GB); 64 sections fit 2 nodes in
# one wave. 4000 m validation sections need about twice the time and memory:
# submit them separately with CORES_PER_TASK=6.
# Reruns skip sections whose H5 is already complete.
#
#SBATCH -J sections
#SBATCH -o sections_%j.o
#SBATCH -e sections_%j.e
#SBATCH -p spr
#SBATCH -N 2
#SBATCH -n 224
#SBATCH -t 12:00:00
#SBATCH -A ECS24003

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CASE_DIR="$(cd "${HPC_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
export SEISKIT_ROOT="${SEISKIT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
MANIFEST="$(realpath "${MANIFEST:?set MANIFEST=sections/*.csv}")"
OUT_DIR="${OUT_DIR:-${SCRATCH}/case7_sections/$(basename "${MANIFEST}" .csv)}"
CORES_PER_TASK="${CORES_PER_TASK:-3}"

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
python "${HPC_DIR}/make_section_tasks.py" --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" > "${TASKS}"
echo "$(date -Is) tasks=$(wc -l < "${TASKS}") nodes=${SLURM_NNODES:-1} cores/task=${CORES_PER_TASK} out=${OUT_DIR}"
[[ -s "${TASKS}" ]] || { echo "nothing to run"; exit 0; }

cd "${OUT_DIR}"
python - "${TASKS}" "${CORES_PER_TASK}" <<'PY'
import sys
import pylauncher

pylauncher.ClassicLauncher(sys.argv[1], cores=int(sys.argv[2]), debug="job")
PY
echo "$(date -Is) done; failures: $(cat "${OUT_DIR}/failures.log" 2>/dev/null | wc -l)"
