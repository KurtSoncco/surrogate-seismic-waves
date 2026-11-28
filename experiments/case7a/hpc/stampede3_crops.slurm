#!/bin/bash
# Five extended Case 7 crops on one Stampede3 SPR node.
#
#   cd ~/surrogate-seismic-waves/experiments/case7a/hpc
#   sbatch stampede3_crops.slurm
#
# Each mesh is 1500 m by 150 m at 1 m (225k quads), smaller than a 2000 m
# section. CORES_PER_TASK=3 reserves memory the way stampede3_sections.slurm
# does. A finished H5 is skipped.
#
#SBATCH -J case7crops
#SBATCH -o case7crops_%j.o
#SBATCH -e case7crops_%j.e
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -t 06:00:00
#SBATCH -A ECS24003

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CASE_DIR="$(cd "${HPC_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
if [[ ! -f "${VENV_PATH}/bin/activate" && -f /work2/09739/kurtsoncco1406/stampede3/seiskit/.venv/bin/activate ]]; then
  VENV_PATH=/work2/09739/kurtsoncco1406/stampede3/seiskit/.venv
fi
export SEISKIT_ROOT="${SEISKIT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
MESHES="$(realpath "${MESHES:-${PROJECT_ROOT}/data/case7a/opensees_crops/meshes}")"
OUT_DIR="${OUT_DIR:-${SCRATCH}/case7_opensees_crops}"
CORES_PER_TASK="${CORES_PER_TASK:-3}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 BASIN_SOLVER="${BASIN_SOLVER:-UmfPack}"

module load python3 pylauncher 2>/dev/null || module load pylauncher
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
mkdir -p "${OUT_DIR}"

TASKS="${OUT_DIR}/tasks_${SLURM_JOB_ID:-local}.txt"
python "${HPC_DIR}/make_crop_tasks.py" --meshes-dir "${MESHES}" --out-dir "${OUT_DIR}" > "${TASKS}"
echo "$(date -Is) tasks=$(wc -l < "${TASKS}") nodes=${SLURM_NNODES:-1} cores/task=${CORES_PER_TASK} out=${OUT_DIR}"
[[ -s "${TASKS}" ]] || { echo "nothing to run"; exit 0; }

cd "${OUT_DIR}"
python - "${TASKS}" "${CORES_PER_TASK}" <<'PY'
import sys
import pylauncher

pylauncher.ClassicLauncher(sys.argv[1], cores=int(sys.argv[2]), debug="job")
PY
echo "$(date -Is) done; failures: $(cat "${OUT_DIR}/failures.log" 2>/dev/null | wc -l)"
