#!/bin/bash
# Score Part 0 leftover ckpts on nested IID probes (6/76, 29/112, 50/148).
#
#   ssh savio
#   cd ~/surrogate-seismic-waves/experiments/DeepONet-Residual
#   sbatch savio_score_probes.sh
#
#SBATCH --job-name=gino_score
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=04:00:00
#SBATCH --output=gino_score.o%j
#SBATCH --error=gino_score.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/home/users/kurtwal98/surrogate-seismic-waves}"
SCRATCH_DATA="${SCRATCH_DATA:-/global/scratch/users/kurtwal98/neural_operator_data}"
EXP_DIR="${PROJECT_ROOT}/experiments/DeepONet-Residual"

export GIFNO_DATA_ROOT="${SCRATCH_DATA}"
export GIFNO_H5_DIR="${SCRATCH_DATA}/h5"
export GIFNO_TF_DIR="${SCRATCH_DATA}/transfer_function"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.2.1 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== score_corner_probes ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
ls -lh "${EXP_DIR}/checkpoints/M7680_xi_field_acf_ft.pt" \
       "${EXP_DIR}/checkpoints/M7680_gno_rh_dilate_ft.pt"

if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun python -u "${EXP_DIR}/response_variability/score_corner_probes.py" --batch-size 4
else
    python -u "${EXP_DIR}/response_variability/score_corner_probes.py" --batch-size 4
fi

echo "Done. ${EXP_DIR}/results/response_variability/eval_bias/part0_probes.csv"
