#!/bin/bash
# Stage 1 then stage 2 from the finished residual seed, pair weight 0.
# Writes under checkpoints/early_width so the live stage-0 curriculum can
# keep its own stage1/ and stage2/ names when it reaches them.
#
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_stage12.sh
#
#SBATCH --job-name=vd_width
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=36:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
INIT="${INIT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/stage0_residual_s0}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/early_width}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
[[ -f "${INIT}/best_model.pt" ]] || { echo "missing ${INIT}/best_model.pt" >&2; exit 1; }
mkdir -p "${OUT}"

echo "=== stage 1 from ${INIT} -> ${OUT}/stage1 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" --encoding metric \
  --stage 1 --target residual --pair-weight 0 --seed 0 \
  --init "${INIT}" --out "${OUT}/stage1" --overwrite

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
  --out "${OUT}/stage2" --overwrite

srun python -u "${VD_DIR}/score_holdout.py" --model-dir "${OUT}/stage2" --cache "${CACHE}"
echo "stage 2 checkpoint ${OUT}/stage2"
