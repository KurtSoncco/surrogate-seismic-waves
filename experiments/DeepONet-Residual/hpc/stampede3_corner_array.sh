#!/bin/bash
# Stampede3 SKX job array for the corner OpenSees campaign (160 + 10).
#
# Fallback when Savio has no seiskit OpenSees env. Pattern from
# seiskit/neural-operator/data/stampede3_single_index.sh.
#
#   MANIFEST=.../corner_is_all.csv sbatch stampede3_corner_array.sh
#
#SBATCH -J corner_ops
#SBATCH -o stampede3_corner_%A_%a.o
#SBATCH -e stampede3_corner_%A_%a.e
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -A ECS24003
#SBATCH --array=0-169

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data}"
RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/corner_is_all.csv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

INDEX="${SLURM_ARRAY_TASK_ID:-${INDEX:-0}}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_BASE="${RUN_BASE:-${SCRATCH:-${PROJECT_ROOT}}/opensees_corner_is}"

export SOBOL_OUTDIR="${SOBOL_OUTDIR:-${RUN_BASE}/raw_runs}"
export SOBOL_H5_DIR="${SOBOL_H5_DIR:-${RUN_BASE}/h5}"
export SOBOL_TIMING_DB="${SOBOL_TIMING_DB:-${RUN_BASE}/sobol_timing.db}"
export SOBOL_H5_LOSSY="${SOBOL_H5_LOSSY:-1}"
export SOBOL_H5_DOWNSAMPLE="${SOBOL_H5_DOWNSAMPLE:-2}"
export SOBOL_MAX_TIME_PER_BATCH="${SOBOL_MAX_TIME_PER_BATCH:-28800}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_BASE}" "${SOBOL_OUTDIR}" "${SOBOL_H5_DIR}"

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: venv not found at ${VENV_PATH}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

if [ ! -r "${RUNNER_PY}" ]; then
  echo "ERROR: runner not readable at ${RUNNER_PY}" >&2
  exit 2
fi
if [ ! -r "${MANIFEST_PATH}" ]; then
  echo "ERROR: manifest not readable at ${MANIFEST_PATH}" >&2
  exit 2
fi

OPENSEES_LIB_DIR="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site

search_paths = []
try:
    search_paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    search_paths.append(site.getusersitepackages())
except Exception:
    pass

for base in search_paths:
    candidate = Path(base) / "openseespylinux" / "lib"
    if candidate.exists():
        print(candidate)
        break
PY
)"
if [ -n "${OPENSEES_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${OPENSEES_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "$(date -Is) | START | Job=${SLURM_JOB_ID:-<local>} Task=${INDEX} Host=$(hostname)" >&2
echo "$(date -Is) | PATHS | MANIFEST=${MANIFEST_PATH} RUN_BASE=${RUN_BASE}" >&2

CMD=(
  "${PYTHON_BIN}" -u "${RUNNER_PY}"
  --manifest-path "${MANIFEST_PATH}"
  --index "${INDEX}"
)
if [ "${FORCE_RERUN}" = "1" ]; then
  CMD+=(--force)
fi
echo "$(date -Is) | EXEC | ${CMD[*]}" >&2
"${CMD[@]}"
echo "$(date -Is) | DONE | Index ${INDEX}" >&2
