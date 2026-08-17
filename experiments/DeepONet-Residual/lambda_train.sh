#!/bin/bash
# Residual GINO on a Lambda Labs GPU (gpu_1x_a10 or A100 is enough).
#
# You start the instance. From the laptop:
#
#   HOST=ubuntu@<LAMBDA_IP>
#   rsync -az --exclude .venv --exclude wandb --exclude '*.pt' \
#     ~/surrogate-seismic-waves/ "$HOST:~/surrogate-seismic-waves/"
#   rsync -az ~/surrogate-seismic-waves/data/gifno_screen/ \
#     "$HOST:~/surrogate-seismic-waves/data/gifno_screen/"
#   rsync -az ~/surrogate-seismic-waves/experiments/DeepONet-Residual/cache/ \
#     "$HOST:~/surrogate-seismic-waves/experiments/DeepONet-Residual/cache/"
#   scp experiments/GIFNO/lambda_secrets.env \
#     "$HOST:~/surrogate-seismic-waves/experiments/GIFNO/"
#
# On the instance (venv + cu128 torch, then tmux):
#
#   ssh "$HOST"
#   cd ~/surrogate-seismic-waves && source .venv/bin/activate
#   tmux new-session -d -s gino \
#     "bash experiments/DeepONet-Residual/lambda_train.sh \
#        --mix M2100 --encoder gno --fno --batch-size 32 \
#        --fno-width 64 --fno-modes 8,32 --fno-layers 4 \
#        --run-name M2100_gino_wide_lambda 2>&1 | tee train_gino.log"
#   tail -f ~/surrogate-seismic-waves/train_gino.log
#
# Pull checkpoints back:
#   rsync -az "$HOST:~/surrogate-seismic-waves/experiments/DeepONet-Residual/checkpoints/*_lambda.pt" \
#     experiments/DeepONet-Residual/checkpoints/
#
# Optional laptop wandb sync of §10 offline runs (needs WANDB_API_KEY):
#   wandb sync experiments/DeepONet-Residual/wandb/offline-run-*

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/surrogate-seismic-waves}"
DATA_ROOT="${GIFNO_DATA_ROOT:-${PROJECT_ROOT}/data/gifno_screen}"
EXP_DIR="${PROJECT_ROOT}/experiments/DeepONet-Residual"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"

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
    export WANDB_MODE="${WANDB_MODE:-online}"
fi

export GIFNO_DATA_ROOT="${DATA_ROOT}"
export WANDB_PROJECT="${WANDB_PROJECT:-deeponet-nscale}"
export WANDB_HOST="${WANDB_HOST:-lambda}"
if [[ -d "${DATA_ROOT}/ood_dipping" ]]; then
    export GIFNO_OOD_DIPPING="${DATA_ROOT}/ood_dipping"
fi
if [[ -d "${DATA_ROOT}/ood_three_layer" ]]; then
    export GIFNO_OOD_THREE_LAYER="${DATA_ROOT}/ood_three_layer"
fi

fail_missing() {
    echo "ERROR: missing required path: $1" >&2
    exit 1
}

[[ -d "${DATA_ROOT}/h5" ]] || fail_missing "${DATA_ROOT}/h5"
[[ -e "${DATA_ROOT}/transfer_function/tf_per_sample.npy" ]] || fail_missing "${DATA_ROOT}/transfer_function/tf_per_sample.npy"
[[ -d "${GIFNO_OOD_DIPPING:-${DATA_ROOT}/ood_dipping}" ]] || fail_missing "${DATA_ROOT}/ood_dipping"
[[ -d "${GIFNO_OOD_THREE_LAYER:-${DATA_ROOT}/ood_three_layer}" ]] || fail_missing "${DATA_ROOT}/ood_three_layer"
[[ -f "${EXP_DIR}/cache/n1000_seed42/r_nom_signed.npy" ]] || fail_missing "${EXP_DIR}/cache/n1000_seed42/r_nom_signed.npy"
[[ -f "${EXP_DIR}/cache/n2000_seed42/r_nom_signed.npy" ]] || fail_missing "${EXP_DIR}/cache/n2000_seed42/r_nom_signed.npy"
if [[ " $* " == *" M2100 "* ]]; then
    [[ -f "${EXP_DIR}/cache/n3000_seed42/r_nom_signed.npy" ]] || fail_missing "${EXP_DIR}/cache/n3000_seed42/r_nom_signed.npy"
fi
if [[ " $* " == *" M7680 "* ]]; then
    [[ -f "${EXP_DIR}/cache/n7680_seed42/r_nom_signed.npy" ]] || fail_missing "${EXP_DIR}/cache/n7680_seed42/r_nom_signed.npy"
fi

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "=== DeepONet-Residual Lambda ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "GIFNO_DATA_ROOT=${GIFNO_DATA_ROOT}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE:-unset}"
echo "WANDB_HOST=${WANDB_HOST}"
echo "WANDB_API_KEY: $([[ -n ${WANDB_API_KEY:-} ]] && echo set || echo missing)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "H5:  $(ls "${DATA_ROOT}/h5"/run_*.h5 2>/dev/null | wc -l) files"
echo "Args: $*"
echo "================================"

python -u "${EXP_DIR}/arch_train.py" "$@"
