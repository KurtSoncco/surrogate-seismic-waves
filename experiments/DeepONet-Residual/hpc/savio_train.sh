#!/bin/bash
# Residual GINO Part 0 fine-tunes on Savio GPU.
#
# Submit from a Savio LOGIN node (ln000), not the DTN:
#   ssh savio
#   cd ~/surrogate-seismic-waves
#   rsync from WSL first (exclude .venv)
#   cd experiments/DeepONet-Residual
#   PART=0a sbatch savio_train.sh
#   PART=0b sbatch savio_train.sh
#   PART=0c sbatch savio_train.sh   # only if sample 50 still below OPS–OPS
#
# Interactive: salloc on savio4_gpu, then PART=0a bash savio_train.sh
#
#SBATCH --job-name=gino_p0
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=24:00:00
#SBATCH --output=gino_p0_%x.o%j
#SBATCH --error=gino_p0_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/home/users/kurtwal98/surrogate-seismic-waves}"
SCRATCH_DATA="${SCRATCH_DATA:-/global/scratch/users/kurtwal98/neural_operator_data}"
EXP_DIR="${PROJECT_ROOT}/experiments/DeepONet-Residual"
INIT_CKPT="${INIT_CKPT:-${EXP_DIR}/checkpoints/M7680_gino_rebal_ft.pt}"
PART="${PART:-0a}"

export GIFNO_DATA_ROOT="${SCRATCH_DATA}"
export GIFNO_H5_DIR="${SCRATCH_DATA}/h5"
export GIFNO_TF_DIR="${SCRATCH_DATA}/transfer_function"

fail_missing() {
    echo "ERROR: missing required path: $1" >&2
    exit 1
}

[[ -d "${GIFNO_H5_DIR}" ]] || fail_missing "${GIFNO_H5_DIR}"
[[ -e "${GIFNO_TF_DIR}/tf_per_sample.npy" ]] || fail_missing "${GIFNO_TF_DIR}/tf_per_sample.npy"
[[ -f "${INIT_CKPT}" ]] || fail_missing "${INIT_CKPT}"
[[ -f "${EXP_DIR}/cache/n7680_seed42/r_nom_signed.npy" ]] || fail_missing "${EXP_DIR}/cache/n7680_seed42/r_nom_signed.npy"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.2.1 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/.venv/bin/activate"

COMMON=(
    --mix M7680
    --encoder gno
    --fno
    --iid-frac 0.34
    --val-monitor three_layer
    --init-ckpt "${INIT_CKPT}"
)

case "${PART}" in
    0a)
        EXTRA=(
            --stoch-layout xi_field_acf
            --freeze-gno
            --run-name M7680_xi_field_acf_ft
        )
        ;;
    0b)
        EXTRA=(
            --gno-rh-dilate
            --encoder-lr 1e-4
            --run-name M7680_gno_rh_dilate_ft
        )
        ;;
    0c)
        EXTRA=(
            --freeze-gno
            --fno-modes 8,32
            --run-name M7680_fno_modes832_ft
        )
        ;;
    *)
        echo "ERROR: PART must be 0a, 0b, or 0c (got ${PART})" >&2
        exit 2
        ;;
esac

echo "=== DeepONet-Residual Savio Part ${PART} ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "H5:  $(ls "${GIFNO_H5_DIR}"/run_*.h5 2>/dev/null | wc -l) files"
echo "Args: ${COMMON[*]} ${EXTRA[*]} $*"
echo "==========================================="

if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun python -u "${EXP_DIR}/arch_train.py" "${COMMON[@]}" "${EXTRA[@]}" "$@"
else
    python -u "${EXP_DIR}/arch_train.py" "${COMMON[@]}" "${EXTRA[@]}" "$@"
fi

echo "Done. Checkpoint: ${EXP_DIR}/checkpoints/ (run-name from PART=${PART})"
