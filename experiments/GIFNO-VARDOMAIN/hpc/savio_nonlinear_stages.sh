#!/bin/bash
# Residual stage 0, then stage 1 and stage 2, with the depth autoencoder and
# the nonlinear spectral decoder. Starts from scratch. The 8-epoch probe
# weights are not the initialization. One A5000, 72 hours.
#
# Submit only after savio_nonlinear_probe.sh has written a passing gate:
#   sbatch --dependency=afterok:<probe_job> experiments/GIFNO-VARDOMAIN/hpc/savio_nonlinear_stages.sh
#
#SBATCH --job-name=vd_nl_stages
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
GATE="${GATE:-/global/scratch/users/kurtwal98/vardomain/checkpoints/nonlinear_probe/probe_gate.json}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/nonlinear}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
[[ -f "${GATE}" ]] || { echo "missing ${GATE}" >&2; exit 1; }
python - "${GATE}" << 'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
if not gate.get("passed"):
    raise SystemExit(f"probe gate has not passed: {gate.get('reasons')}")
PY

mkdir -p "${OUT}"
FLAGS=(
  --depth-ae --readout nonlinear --recon-weight 0.1
  --local-min-wavelength 0 --hfp-kernel 4 --hfp-stride 4
)

echo "=== stage 0 from scratch -> ${OUT}/stage0_residual_s0 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" --encoding metric \
  --stage 0 --target residual --pair-weight 0 --seed 0 \
  --out "${OUT}/stage0_residual_s0" --overwrite \
  "${FLAGS[@]}"

echo "=== stage 1 from stage 0 -> ${OUT}/stage1 ==="
srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" --encoding metric \
  --stage 1 --target residual --pair-weight 0 --seed 0 \
  --init "${OUT}/stage0_residual_s0" --out "${OUT}/stage1" --overwrite \
  "${FLAGS[@]}"

IID_BASE="$(python - << PY
import csv
from pathlib import Path
rows = list(csv.reader(Path("${OUT}/stage1/history.csv").open()))
body = [r for r in rows if r and r[0] != "epoch"]
print(min(float(r[2]) for r in body))
PY
)"
echo "stage 1 IID baseline ${IID_BASE}"

echo "=== stage 2 -> ${OUT}/stage2 ==="
srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" --encoding metric \
  --stage 2 --target residual --pair-weight 0 --seed 0 \
  --init "${OUT}/stage1" --iid-baseline "${IID_BASE}" \
  --out "${OUT}/stage2" --overwrite \
  "${FLAGS[@]}"

srun python -u "${VD_DIR}/score_holdout.py" --model-dir "${OUT}/stage2" --cache "${CACHE}"
echo "stage 2 checkpoint ${OUT}/stage2"
