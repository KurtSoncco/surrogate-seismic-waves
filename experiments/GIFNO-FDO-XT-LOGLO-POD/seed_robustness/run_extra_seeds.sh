#!/usr/bin/env bash
# Run supplemental OpenSees jobs (replicate_id 30..49) for seed-robustness study.
#
# Usage:
#   ./seed_robustness/run_extra_seeds.sh --sample-id 0
#   ./seed_robustness/run_extra_seeds.sh --sample-id 0 --dry-run
#
# Environment:
#   SEISKIT_DATA_DIR   Path to neural-operator/data (default: ~/seiskit/neural-operator/data)
#   SOBOL_H5_DIR       Output H5 directory (default: checkpoints/seed_robustness/h5)
#   SEISKIT_VENV       Python venv with openseespy (default: ~/seiskit/.venv)
#   MANIFEST_50        Full 12800-row manifest (default: checkpoints/seed_robustness/sobol_manifest_50.csv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"

SEISKIT_DATA_DIR="${SEISKIT_DATA_DIR:-${HOME}/seiskit/neural-operator/data}"
SOBOL_H5_DIR="${SOBOL_H5_DIR:-${REPO_ROOT}/checkpoints/seed_robustness/h5}"
SEISKIT_VENV="${SEISKIT_VENV:-${HOME}/seiskit/.venv}"
MANIFEST_OUT="${REPO_ROOT}/checkpoints/seed_robustness"
MANIFEST_50="${MANIFEST_50:-${MANIFEST_OUT}/sobol_manifest_50.csv}"
SEEDS_PER_SAMPLE="${SEEDS_PER_SAMPLE:-50}"
SAMPLE_ID="${SAMPLE_ID:-0}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-id) SAMPLE_ID="$2"; shift 2 ;;
    --h5-dir) SOBOL_H5_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

PYTHON="${SEISKIT_VENV}/bin/python"
RUNNER="${SEISKIT_DATA_DIR}/run_experiment.py"

mkdir -p "${SOBOL_H5_DIR}" "${MANIFEST_OUT}"

echo "[run_extra_seeds] sample_id=${SAMPLE_ID}  SOBOL_H5_DIR=${SOBOL_H5_DIR}"

cd "${EXP_DIR}"
uv run python seed_robustness/manifest_extra_seeds.py \
  --sample-id "${SAMPLE_ID}" \
  --seeds-per-sample "${SEEDS_PER_SAMPLE}" \
  --out-dir "${MANIFEST_OUT}" \
  --extra-only \
  --print-indices

if [[ ! -f "${MANIFEST_50}" ]]; then
  echo "[manifest] Creating ${MANIFEST_50} (256 x 50 = 12800 rows) ..."
  (cd "${SEISKIT_DATA_DIR}" && "${PYTHON}" -c "
import sys; sys.path.insert(0, '.')
from sobol import build_manifest, write_manifest_csv
from pathlib import Path
out = Path('${MANIFEST_50}')
out.parent.mkdir(parents=True, exist_ok=True)
write_manifest_csv(out, build_manifest(seeds_per_sample=50))
print('wrote', out, 'rows', 12800)
")
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: seiskit venv not found at ${PYTHON}" >&2
  exit 1
fi

export SOBOL_H5_DIR
export PYTHONPATH="${SEISKIT_DATA_DIR}:${HOME}/seiskit:${PYTHONPATH:-}"

START=$((SAMPLE_ID * SEEDS_PER_SAMPLE + 30))
END=$((SAMPLE_ID * SEEDS_PER_SAMPLE + 50))

echo "[run_extra_seeds] Running indices ${START}..$((END - 1)) via run_experiment.py"
echo "[run_extra_seeds] Manifest: ${MANIFEST_50}"

for IDX in $(seq "${START}" $((END - 1))); do
  H5="${SOBOL_H5_DIR}/run_${IDX}.h5"
  if [[ -f "${H5}" ]]; then
    echo "[skip] index ${IDX} — exists"
    continue
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would run index ${IDX}"
    continue
  fi
  echo "[run] index ${IDX} ..."
  "${PYTHON}" -u "${RUNNER}" --manifest-path "${MANIFEST_50}" --index "${IDX}"
done

echo "[run_extra_seeds] Done. Export SEED_ROBUSTNESS_EXTRA_H5_DIR=${SOBOL_H5_DIR} for analysis."
