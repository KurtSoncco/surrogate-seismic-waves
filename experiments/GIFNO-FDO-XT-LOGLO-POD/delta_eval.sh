#!/bin/bash
# Post-hoc test evaluation for a saved LOGLO-POD checkpoint (metrics + W&B plots).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD
#   source ../GIFNO/delta_env.sh
#
#   export SWEEP_VARIANT=loglo_pod_ref
#   export SWEEP_LIMIT=500
#   sbatch delta_eval.sh --limit 500 --wandb-run-id <id>
#
# Full-dataset eval (omit --limit):
#   export SWEEP_VARIANT=loglo_pod_ref
#   export SWEEP_TAG=full
#   sbatch delta_eval.sh --wandb-run-name sweep_loglo_pod_ref_full_eval

#SBATCH --job-name=loglo_pod_eval
#SBATCH --account=bgpu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=loglo_pod_eval.o%j
#SBATCH --error=loglo_pod_eval.e%j

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

GIFNO_DIR="$(cd "${SCRIPT_DIR}/../GIFNO" && pwd)"
# shellcheck source=../GIFNO/delta_env.sh
[[ -f "${GIFNO_DIR}/delta_env.sh" ]] && source "${GIFNO_DIR}/delta_env.sh"
# shellcheck source=../GIFNO/delta_paths.sh
source "${GIFNO_DIR}/delta_paths.sh"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v module &>/dev/null; then
    module reset 2>/dev/null || true
fi

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(resolve_gifno_data_root)"
LOGLO_DIR="${SCRIPT_DIR}"

SECRETS_FILE="${GIFNO_DIR}/lambda_secrets.env"
if [[ -f "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    set +a
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="${WANDB_API_KEY//$'\r'/}"
    WANDB_API_KEY="${WANDB_API_KEY#"${WANDB_API_KEY%%[![:space:]]*}"}"
    WANDB_API_KEY="${WANDB_API_KEY%"${WANDB_API_KEY##*[![:space:]]}"}"
    export WANDB_API_KEY
fi

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_H5_DIR="${GIFNO_H5_DIR:-${DATA_ROOT}/h5}"
export GIFNO_TF_DIR="${GIFNO_TF_DIR:-${DATA_ROOT}/transfer_function}"

if [[ -n "${SWEEP_VARIANT:-}" ]]; then
    if [[ -n "${SWEEP_TAG:-}" ]]; then
        :
    elif [[ "${SWEEP_FULL:-0}" == "1" ]]; then
        SWEEP_TAG="full"
    else
        SWEEP_TAG="n${SWEEP_LIMIT:-500}"
    fi
    export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-${GIFNO_TF_DIR}/models/fdo_xt_loglo_pod/sweep/${SWEEP_TAG}/${SWEEP_VARIANT}}"
    export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-${GIFNO_TF_DIR}/results/fdo_xt_loglo_pod/sweep/${SWEEP_TAG}/${SWEEP_VARIANT}}"
fi

export GIFNO_MODEL_DIR="${GIFNO_MODEL_DIR:-${GIFNO_TF_DIR}/models/fdo_xt_loglo_pod}"
export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-${GIFNO_TF_DIR}/results/fdo_xt_loglo_pod}"

CKPT="${GIFNO_MODEL_DIR}/best_model.pt"
for req in \
    "${GIFNO_H5_DIR}" \
    "${GIFNO_TF_DIR}/tf_per_sample.npy" \
    "${GIFNO_TF_DIR}/manifest.csv" \
    "${GIFNO_TF_DIR}/recorder_x_idx.npy" \
    "${GIFNO_TF_DIR}/pod_modes.npy" \
    "${GIFNO_TF_DIR}/pod_mean.npy" \
    "${CKPT}"; do
    if [[ ! -e "${req}" ]]; then
        echo "ERROR: missing required path: ${req}" >&2
        exit 1
    fi
done

mkdir -p "${GIFNO_RESULTS_DIR}"

load_delta_python() {
    if module is-loaded python 2>/dev/null; then
        return 0
    fi
    module load gcc python 2>/dev/null || module load python
}
load_delta_python

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== GIFNO-FDO-XT-LOGLO-POD eval on Delta ==="
echo "WANDB_PROJECT=gifno_fdo_xt_loglo_pod"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "GIFNO_MODEL_DIR=${GIFNO_MODEL_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Args: $*"
echo "============================================"

EVAL_ARGS=("$@")
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    has_run_id=false
    for ((i = 0; i < ${#EVAL_ARGS[@]}; i++)); do
        if [[ "${EVAL_ARGS[i]}" == "--wandb-run-id" ]]; then
            has_run_id=true
            break
        fi
    done
    if [[ "${has_run_id}" == "false" ]]; then
        EVAL_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
    fi
fi

python -u "${LOGLO_DIR}/eval_checkpoint.py" "${EVAL_ARGS[@]}"

echo "Done. Metrics/plots: ${GIFNO_RESULTS_DIR}/"
