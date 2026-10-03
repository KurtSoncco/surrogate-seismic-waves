#!/bin/bash
# Savio CPU job array for the corner OpenSees campaign (160 + 10).
#
# Prefer this if ~/seiskit has a working OpenSees venv. Else use
# stampede3_corner_array.sh.
#
#   mkdir -p logs
#   MANIFEST=.../corner_is_all.csv sbatch savio_corner_array.sh
#
#SBATCH --job-name=corner_ops
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-169%20
#SBATCH --output=logs/corner_ops_%A_%a.out
#SBATCH --error=logs/corner_ops_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/seiskit}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/neural-operator/data}"
RUNNER_PY="${RUNNER_PY:-${SCRIPT_DIR}/run_experiment.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${HOME}/surrogate-seismic-waves/experiments/DeepONet-Residual/results/response_variability/eval_bias/corner_is_all.csv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

INDEX="${SLURM_ARRAY_TASK_ID:-${INDEX:-0}}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SCRATCH_BASE="${SCRATCH_BASE:-/global/scratch/users/${USER}/neural_operator_data/corner_is}"
RUN_BASE="${RUN_BASE:-${SCRATCH_BASE}}"

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

mkdir -p logs "${RUN_BASE}" "${SOBOL_OUTDIR}" "${SOBOL_H5_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  module purge || true
  module load gcc/13.2.0 openblas/0.3.24 || true
fi

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "ERROR: seiskit venv not found at ${VENV_PATH}" >&2
  echo "Use stampede3_corner_array.sh if OpenSees lives on TACC." >&2
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
