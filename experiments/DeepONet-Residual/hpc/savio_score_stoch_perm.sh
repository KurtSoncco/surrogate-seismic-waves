#!/bin/bash
# Permutation importance of ξ vs rH/aHV/CoV on leftover GINO + Mscale ckpts.
#
#   sbatch savio_score_stoch_perm.sh
#
#SBATCH --job-name=stoch_perm
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --exclude=n0215.savio3
#SBATCH --time=02:00:00
#SBATCH --output=gino_stoch_perm.o%j
#SBATCH --error=gino_stoch_perm.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
SCRATCH_DATA="${SCRATCH_DATA:-/global/scratch/users/kurtwal98/neural_operator_data}"
EXP_DIR="${PROJECT_ROOT}/experiments/DeepONet-Residual"

export GIFNO_DATA_ROOT="${SCRATCH_DATA}"
export GIFNO_H5_DIR="${SCRATCH_DATA}/h5"
export GIFNO_TF_DIR="${SCRATCH_DATA}/transfer_function"
if [[ -d "${SCRATCH_DATA}/ood_dipping" ]]; then
    export GIFNO_OOD_DIPPING="${SCRATCH_DATA}/ood_dipping"
fi
if [[ -d "${SCRATCH_DATA}/ood_three_layer" ]]; then
    export GIFNO_OOD_THREE_LAYER="${SCRATCH_DATA}/ood_three_layer"
fi

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== stoch permutation (Mscale + GINO) ==="
echo "HOST=$(hostname)  GPU=$(srun nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo cpu)"
echo "GIFNO_TF_DIR=${GIFNO_TF_DIR}"
srun ls -lh \
    "${EXP_DIR}/checkpoints/iid2000_gino_fno.pt" \
    "${EXP_DIR}/checkpoints/iid2000_gino_fno_mscaleT4.pt" \
    "${EXP_DIR}/checkpoints/iid2000_gino_fno_mscaleT4B4.pt" \
    "${EXP_DIR}/checkpoints/iid2000_gino_fno_ff5.pt" \
    "${EXP_DIR}/checkpoints/m1400_gino_fno_mscaleT4B4.pt" \
    "${EXP_DIR}/checkpoints/M7680_gino_rebal_ft.pt"

srun python -u "${EXP_DIR}/scoring/score_stoch_permutation.py" --batch-size 16 \
    --out "${EXP_DIR}/results/arch_train/stoch_permutation.json"
echo "=== done $(date -Is) ==="
