#!/usr/bin/env bash
# From-scratch interpolating LOGLO on a Lambda A10 (collocation head, mix of everybody).
#
#   ssh ubuntu@146.235.208.227
#   bash ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD/lambda_dual_head.sh setup
#   bash ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD/lambda_dual_head.sh check
#   bash ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD/lambda_dual_head.sh launch
#
# launch starts tmux session `loglo_mix`. TARGETS=outcrop residual, SEEDS=0 first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${PROJECT_ROOT}/.venv"
DATA="${PROJECT_ROOT}/data"
SCREEN="${DATA}/gifno_screen"
SECRETS_FILE="${PROJECT_ROOT}/experiments/GIFNO/lambda_secrets.env"

export IID_DIR="${IID_DIR:-${SCREEN}/transfer_function}"
export DIP_DIR="${DIP_DIR:-${SCREEN}/ood_dipping}"
export TL_DIR="${TL_DIR:-${SCREEN}/ood_three_layer}"
export CORNER_DIR="${CORNER_DIR:-${SCREEN}/corner_is}"
export SECTION_DIRS="${SECTION_DIRS:-${DATA}/case7a/stampede_round0/windows_r0}"
export DATA_OUT="${DATA_OUT:-${SCREEN}/outcrop_tf}"
export CK_OUT="${CK_OUT:-${PROJECT_ROOT}/checkpoints/case7_generalization}"
export TARGETS="${TARGETS:-outcrop residual}"
export SEEDS="${SEEDS:-0}"
export STAGE="${STAGE:-all}"
export PYTHON="${PYTHON:-${VENV}/bin/python}"

export GIFNO_POD_LAYOUT="${GIFNO_POD_LAYOUT:-shared}"
# A10 23 GiB: pair flatten doubles the batch. 8 used 1.7 GiB / 69% util;
# 64 → 128 windows, ~14 GiB, still clear of OOM at 96/20.8 GiB.
export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-64}"
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-true}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-8}"
export GIFNO_NUM_EPOCHS="${GIFNO_NUM_EPOCHS:-300}"
export GIFNO_WINDOW_SHIFT_MAX="${GIFNO_WINDOW_SHIFT_MAX:-6}"
export GIFNO_WINDOW_CONSISTENCY_WEIGHT="${GIFNO_WINDOW_CONSISTENCY_WEIGHT:-0.1}"
export GIFNO_SELECTION_METRIC="${GIFNO_SELECTION_METRIC:-pearson_trusted}"
export GIFNO_NZ_MAX="${GIFNO_NZ_MAX:-160}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-gifno_case7_generalization}"

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
}

setup_venv() {
  if [[ ! -x "${VENV}/bin/python" ]]; then
    if ! python3 -m venv --system-site-packages "${VENV}"; then
      sudo apt-get update
      sudo apt-get install -y python3-venv python3-dev
      python3 -m venv --system-site-packages "${VENV}"
    fi
  fi
  activate_venv
  python -m pip install -U pip
  python -m pip install \
    'numpy>=1.26,<2' \
    scipy matplotlib pandas h5py hdf5plugin tqdm wandb \
    pyarrow scikit-learn \
    'neuraloperator>=2.0.0'
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
}

check_data() {
  local missing=0
  require() {
    if [[ ! -e "$1" ]]; then
      echo "missing $1" >&2
      missing=1
    fi
  }
  require "${IID_DIR}/tf_outcrop_per_sample.npy"
  require "${IID_DIR}/outcrop_valid.npy"
  require "${DIP_DIR}/tf_outcrop_per_sample.npy"
  require "${TL_DIR}/tf_outcrop_per_sample.npy"
  require "${CORNER_DIR}/tf_outcrop_per_sample.npy"
  require "${SECTION_DIRS}/tf_outcrop_per_sample.npy"
  require "${SECTION_DIRS}/manifest.csv"
  require "${SCREEN}/h5/run_0.h5"
  require "${SCREEN}/h5/run_7679.h5"
  require "${DIP_DIR}/h5/run_0.h5"
  require "${DATA}/case7a/stampede_round0/h5/sec_0000.h5"
  if [[ "$missing" -ne 0 ]]; then
    echo "data check failed" >&2
    exit 1
  fi
  echo "data ok"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

load_secrets() {
  if [[ -f "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    set +a
    export WANDB_MODE="${WANDB_MODE:-online}"
  fi
}

run_train() {
  activate_venv
  load_secrets
  check_data
  mkdir -p "${DATA_OUT}" "${CK_OUT}"
  cd "${SCRIPT_DIR}"
  echo "=== dual-head from scratch TARGETS=${TARGETS} SEEDS=${SEEDS} STAGE=${STAGE} ==="
  echo "IID_DIR=${IID_DIR}"
  echo "SECTION_DIRS=${SECTION_DIRS}"
  echo "DATA_OUT=${DATA_OUT} CK_OUT=${CK_OUT}"
  bash "${SCRIPT_DIR}/run_dual_head.sh"
}

cmd="${1:-launch}"
case "$cmd" in
  setup) setup_venv ;;
  check) check_data ;;
  run) run_train ;;
  launch)
    setup_venv
    tmux has-session -t loglo_mix 2>/dev/null && { echo "tmux session loglo_mix already exists"; exit 1; }
    LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/train_dual_head.log}"
    tmux new-session -d -s loglo_mix \
      "bash ${SCRIPT_DIR}/lambda_dual_head.sh run 2>&1 | tee -a ${LOG_FILE}"
    echo "started tmux loglo_mix; tail -f ${LOG_FILE}"
    ;;
  *)
    echo "usage: $0 {setup|check|run|launch}" >&2
    exit 1
    ;;
esac
