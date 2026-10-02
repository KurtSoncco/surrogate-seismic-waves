#!/bin/bash
# Spatial-query kernel GNO on Savio (Phases 0–2).
#
# Submit from a Savio LOGIN node:
#   cd experiments/DeepONet-Residual
#   PART=spatial_control sbatch savio_spatial_query.sh
#   PART=spatial_even    sbatch savio_spatial_query.sh
#   PART=spatial_interior sbatch savio_spatial_query.sh
#   PART=spatial_score   sbatch savio_spatial_query.sh   # after the three trains
#
# Does not overwrite checkpoints/M7680_gino_rebal_ft.pt. Promote a new leftover
# only if nested 21-station ship gates pass AND hold-out beats interpolate-p.
#
#SBATCH --job-name=gino_spatial
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=24:00:00
#SBATCH --output=gino_spatial_%x.o%j
#SBATCH --error=gino_spatial_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/home/users/kurtwal98/surrogate-seismic-waves}"
SCRATCH_DATA="${SCRATCH_DATA:-/global/scratch/users/kurtwal98/neural_operator_data}"
EXP_DIR="${PROJECT_ROOT}/experiments/DeepONet-Residual"
INIT_CKPT="${INIT_CKPT:-${EXP_DIR}/checkpoints/M7680_gino_rebal_ft.pt}"
PART="${PART:-spatial_control}"

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

run_py() {
    if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun python -u "$@"
    else
        python -u "$@"
    fi
}

echo "=== DeepONet-Residual Savio spatial ${PART} ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "==============================================="

echo "[spatial] stride-5 support fields from existing H5 (no OpenSees)"
run_py "${EXP_DIR}/residual_signed.py" --cache-tag n7680_seed42 --support-fields
run_py "${EXP_DIR}/residual_signed.py" --cache-tag n1000_seed42 --slice-support-from n7680_seed42
run_py "${EXP_DIR}/residual_signed.py" --cache-tag n2000_seed42 --slice-support-from n7680_seed42 || true
run_py "${EXP_DIR}/residual_signed.py" --cache-tag n3000_seed42 --slice-support-from n7680_seed42 || true
run_py "${EXP_DIR}/ood_signed_cache.py" --support-fields

COMMON=(
    --mix M7680
    --encoder kernel
    --iid-frac 0.34
    --val-monitor three_layer
    --init-ckpt "${INIT_CKPT}"
    --support-stride 5
    --kernel-k 2
)

case "${PART}" in
    spatial_control)
        EXTRA=(
            --query-split all
            --run-name M7680_kernel_spatial_all
        )
        ;;
    spatial_even)
        EXTRA=(
            --query-split even
            --run-name M7680_kernel_spatial_even
        )
        ;;
    spatial_interior)
        EXTRA=(
            --query-split interior
            --run-name M7680_kernel_spatial_interior
        )
        ;;
    spatial_score)
        echo "[spatial] hold-out vs interpolate-p + nested 21-station gates"
        run_py "${EXP_DIR}/eval_spatial_query.py" \
            --ckpt "${EXP_DIR}/checkpoints/M7680_kernel_spatial_even.pt" \
            --ship-ckpt "${INIT_CKPT}" \
            --hold-out odd
        run_py "${EXP_DIR}/eval_spatial_query.py" \
            --ckpt "${EXP_DIR}/checkpoints/M7680_kernel_spatial_interior.pt" \
            --ship-ckpt "${INIT_CKPT}" \
            --hold-out edge
        run_py "${EXP_DIR}/score_ship_gates.py" \
            "${EXP_DIR}/results/arch_train/M7680_kernel_spatial_all.json" \
            --out "${EXP_DIR}/results/arch_train/spatial_query_ship_gates.json"
        echo "Done scoring. Do not replace ${INIT_CKPT} unless both gates pass."
        exit 0
        ;;
    *)
        echo "ERROR: PART must be spatial_control, spatial_even, spatial_interior, or spatial_score (got ${PART})" >&2
        exit 2
        ;;
esac

echo "Args: ${COMMON[*]} ${EXTRA[*]} $*"
run_py "${EXP_DIR}/arch_train.py" "${COMMON[@]}" "${EXTRA[@]}" "$@"

echo "Done. Checkpoint: ${EXP_DIR}/checkpoints/ (run-name from PART=${PART})"
echo "Nested 21-station tests are in the run JSON. Spatial hold-out: PART=spatial_score."
