#!/bin/bash
# Stage 2 only, from the finished early_width stage 1, allowing a 50% IID rise.
# The 10% guard stopped the first stage 2 after one epoch. This does not touch
# the live stage-0 curriculum or that stopped checkpoint.
#
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_stage2_rise.sh
#
#SBATCH --job-name=vd_rise50
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=12:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
INIT="${INIT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/early_width/stage1}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/early_width/stage2_rise50}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
[[ -f "${INIT}/best_model.pt" ]] || { echo "missing ${INIT}/best_model.pt" >&2; exit 1; }
[[ -f "${INIT}/history.csv" ]] || { echo "missing ${INIT}/history.csv" >&2; exit 1; }
mkdir -p "${OUT}"

IID_BASE="$(python - << PY
import csv
from pathlib import Path
rows = list(csv.reader(Path("${INIT}/history.csv").open()))
body = [r for r in rows if r and r[0] != "epoch"]
print(min(float(r[2]) for r in body))
PY
)"
echo "=== stage 2 rise 50% from ${INIT} -> ${OUT} ==="
echo "stage 1 IID baseline ${IID_BASE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" --encoding metric \
  --stage 2 --target residual --pair-weight 0 --seed 0 \
  --init "${INIT}" --iid-baseline "${IID_BASE}" --iid-rise 0.5 \
  --out "${OUT}" --overwrite

srun python -u "${VD_DIR}/score_holdout.py" --model-dir "${OUT}" --cache "${CACHE}"
echo "stage 2 checkpoint ${OUT}"
