#!/bin/bash
# 150 m manifold sections on Stampede3 SKX. One thread per solve. 32 tasks
# share a 192 GB node so a 3000 x 150 m padded mesh stays under the memory.
# Reruns skip an H5 that is already complete.
#
#   cd experiments/GIFNO-VARDOMAIN/hpc
#   MANIFEST=../sections/manifold150.csv sbatch stampede3_manifold.slurm
#
#SBATCH -J manifold150
#SBATCH -o manifold150_%j.o
#SBATCH -e manifold150_%j.e
#SBATCH -p skx
#SBATCH -N 8
#SBATCH -n 256
#SBATCH -t 40:00:00
#SBATCH -A TG-CIV260015

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VD_DIR="$(cd "${HPC_DIR}/.." && pwd)"
CASE_DIR="$(cd "${VD_DIR}/../case7a" && pwd)"
CASE_HPC="${CASE_DIR}/hpc"
PROJECT_ROOT="$(cd "${VD_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
export SEISKIT_ROOT="${SEISKIT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
MANIFEST="$(realpath "${MANIFEST:?set MANIFEST to manifold150.csv}")"
OUT_DIR="${OUT_DIR:-${SCRATCH}/vardomain/manifold150}"
CORES_PER_TASK="${CORES_PER_TASK:-1}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 BASIN_SOLVER="${BASIN_SOLVER:-UmfPack}"

module load python3 pylauncher 2>/dev/null || module load pylauncher
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
mkdir -p "${OUT_DIR}"

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
