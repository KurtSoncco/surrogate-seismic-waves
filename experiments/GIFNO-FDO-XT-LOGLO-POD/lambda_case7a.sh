#!/usr/bin/env bash
# Case 7a on a Lambda GPU box: basin OpenSees, then a log-POD mix fine-tune.
#
# The instance in use is ubuntu@152.67.232.110 (NVIDIA A10, 30 vCPUs, 222 GB).
# From the laptop, after rsync:
#
#   ssh ubuntu@152.67.232.110 \
#     'bash ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD/lambda_case7a.sh'
#
# With no arguments this starts a tmux session named case7a. The session probes
# one column, sizes a process pool from that column's RSS, finishes the 256
# train rows, builds transfer functions and the mix, and fine-tunes on the GPU.
#
#   bash lambda_case7a.sh setup
#   bash lambda_case7a.sh check
#   bash lambda_case7a.sh run
#   bash lambda_case7a.sh launch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASE_DIR="${PROJECT_ROOT}/experiments/case7a"
SEISKIT_ROOT="${SEISKIT_ROOT:-${HOME}/seiskit}"
VENV="${PROJECT_ROOT}/.venv"
DATA="${PROJECT_ROOT}/data"
H5_DIR="${DATA}/gifno_h5"
TF_DIR="${DATA}/gifno_screen/transfer_function"
SCREEN="${DATA}/gifno_screen"
BASIN="${DATA}/basin_ensemble"
MIX="${PROJECT_ROOT}/checkpoints/mix_finetune_data"
MODEL_DIR="${PROJECT_ROOT}/checkpoints/logpod_mix_finetune"
INIT_CKPT="${PROJECT_ROOT}/checkpoints/logpod_full7680/best_model.pt"
SECRETS_FILE="${PROJECT_ROOT}/experiments/GIFNO/lambda_secrets.env"
PROBE_LOG="${BASIN}/probe.log"
POOL_LOG="${BASIN}/queue.log"

export SEISKIT_ROOT
export OPENSEES_ROOT="${SEISKIT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
  # System torch and matplotlib on the Lambda image are built for NumPy 1.x.
  python -m pip install \
    'numpy>=1.26,<2' \
    openseespy scipy matplotlib pandas h5py hdf5plugin tqdm wandb \
    pyarrow statsmodels scikit-learn numba joblib seaborn plotly \
    'neuraloperator>=2.0.0'
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import openseespy.opensees as ops; print('openseespy', ops.__file__)"
}

check_data() {
  local missing=0
  require() {
    if [[ ! -e "$1" ]]; then
      echo "missing $1" >&2
      missing=1
    fi
  }
  require "${H5_DIR}"
  require "${TF_DIR}/tf_per_sample.npy"
  require "${TF_DIR}/manifest.csv"
  require "${TF_DIR}/freq.npy"
  require "${TF_DIR}/recorder_x_idx.npy"
  require "${SCREEN}/ood_dipping/manifest.csv"
  require "${SCREEN}/ood_dipping/h5"
  require "${SCREEN}/ood_three_layer/manifest.csv"
  require "${SCREEN}/ood_three_layer/h5"
  require "${INIT_CKPT}"
  require "${PROJECT_ROOT}/checkpoints/logpod_full7680/pod_modes.npy"
  require "${SEISKIT_ROOT}/seiskit"
  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
  local n_h5
  n_h5="$(find "${H5_DIR}" -maxdepth 1 -name 'run_*.h5' | wc -l)"
  echo "gifno h5 ${n_h5}"
  if [[ "${n_h5}" -lt 7680 ]]; then
    echo "expected 7680 training H5 files, found ${n_h5}" >&2
    exit 1
  fi
  echo "check ok"
}

_rss_kb() {
  local pid="$1"
  awk '/VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || echo 0
}

_choose_workers() {
  local rss_kb="$1"
  local ram_kb
  ram_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  activate_venv
  (
    cd "${CASE_DIR}"
    python -c "import run_basin_ensemble as q; print(q.choose_workers(${rss_kb}/1024/1024, ${ram_kb}/1024/1024))"
  )
}

run_pipeline() {
  check_data
  setup_venv
  activate_venv
  mkdir -p "${BASIN}/h5" "${MIX}" "${MODEL_DIR}"
  cd "${CASE_DIR}"

  echo "=== probe basin_0000 ==="
  python -u run_basin_ensemble.py --index 0 --out-dir "${BASIN}" >"${PROBE_LOG}" 2>&1 &
  local probe_pid=$!
  local rss_max=0
  local rss=0
  local deadline=$((SECONDS + 2700))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${probe_pid}" 2>/dev/null; then
      break
    fi
    rss="$(_rss_kb "${probe_pid}")"
    if [[ -n "${rss}" && "${rss}" -gt "${rss_max}" ]]; then
      rss_max="${rss}"
    fi
    if grep -q "Progress:" "${PROBE_LOG}"; then
      sleep 60
      rss="$(_rss_kb "${probe_pid}")"
      if [[ -n "${rss}" && "${rss}" -gt "${rss_max}" ]]; then
        rss_max="${rss}"
      fi
      break
    fi
    sleep 15
  done

  local total
  total="$(_choose_workers "${rss_max}")"
  echo "probe rss_max_kb=${rss_max} workers=${total}"

  local pool_pid=""
  if kill -0 "${probe_pid}" 2>/dev/null && [[ "${total}" -gt 1 ]]; then
    echo "=== pool start=1 workers=$((total - 1)) ==="
    python -u run_basin_ensemble.py \
      --start 1 --workers "$((total - 1))" --out-dir "${BASIN}" \
      >"${POOL_LOG}" 2>&1 &
    pool_pid=$!
  fi

  set +e
  wait "${probe_pid}"
  local probe_rc=$?
  local pool_rc=0
  if [[ -n "${pool_pid}" ]]; then
    wait "${pool_pid}"
    pool_rc=$?
  fi
  set -e
  echo "probe_rc=${probe_rc} pool_rc=${pool_rc}"
  if [[ ! -f "${BASIN}/h5/run_0.h5" ]] && ! grep -q "Progress:" "${PROBE_LOG}"; then
    echo "probe failed before the first step" >&2
    tail -n 40 "${PROBE_LOG}" >&2
    exit 1
  fi

  echo "=== catch-up workers=${total} ==="
  python -u run_basin_ensemble.py --workers "${total}" --out-dir "${BASIN}" \
    >>"${POOL_LOG}" 2>&1

  local n_h5
  n_h5="$(find "${BASIN}/h5" -maxdepth 1 -name 'run_*.h5' | wc -l)"
  echo "basin h5 ${n_h5}"
  if [[ "${n_h5}" -lt 1 ]]; then
    echo "no basin H5 files were written" >&2
    exit 1
  fi

  if [[ ! -f "${BASIN}/tf_per_sample.npy" ]]; then
    echo "=== transfer functions ==="
    python -u "${PROJECT_ROOT}/experiments/GIFNO/preprocess/compute_transfer_function.py" \
      --results-dir "${BASIN}/h5" \
      --output-dir "${BASIN}"
  fi

  if [[ ! -f "${MIX}/mix_tf_per_sample.npy" ]]; then
    echo "=== mix ==="
    (
      export GIFNO_TF_DIR="${TF_DIR}"
      export GIFNO_H5_DIR="${H5_DIR}"
      cd "${SCRIPT_DIR}"
      python -u prepare_ood_mix_data.py \
        --box-root "${SCREEN}" \
        --basin-dir "${BASIN}" \
        --out-dir "${MIX}" \
        --ood-scores-root "${PROJECT_ROOT}/checkpoints/ood_campaign_scores"
    )
  fi
  ln -sfn mix_tf_per_sample.npy "${MIX}/tf_per_sample.npy"
  ln -sfn mix_manifest.csv "${MIX}/manifest.csv"

  echo "=== log-POD mix fine-tune ==="
  # Absolute paths in the mix manifest point at iid, ood, and basin H5s.
  # GIFNO_H5_DIR would remap every run_*.h5 onto the iid directory.
  unset GIFNO_H5_DIR || true
  export GIFNO_DATA_ROOT="${SCREEN}"
  export GIFNO_TF_DIR="${MIX}"
  export GIFNO_MODEL_DIR="${MODEL_DIR}"
  export GIFNO_RESULTS_DIR="${MODEL_DIR}/results"
  export GIFNO_POD_SPACE=log
  export GIFNO_LOG_TF_LOSS=true
  export GIFNO_REINIT_POD_BRANCH=true
  export GIFNO_INIT_CHECKPOINT="${INIT_CKPT}"
  export GIFNO_LATENT_CHANNELS=128
  export GIFNO_POD_NUM_MODES=64
  export GIFNO_NUM_FNO_LAYERS=5
  export GIFNO_DEEPONET_LATENT_DIM=128
  export GIFNO_LOSS_RADIAL_WEIGHT=0.25
  export GIFNO_LOGLO_PATCH_SIZE=16,20
  export GIFNO_LOGLO_HF_NOISE_ALPHA=0.025
  export GIFNO_BAND_CURRICULUM=true
  export GIFNO_BAND_CURRICULUM_MODE=convergence
  export GIFNO_LOSS_BAND_BALANCED_WEIGHT=0.5
  export GIFNO_BAND_CURRICULUM_LR_RESTART=true
  export GIFNO_EARLY_STOP_PATIENCE=80
  export GIFNO_LR_SCHED_PATIENCE=30
  export GIFNO_NUM_EPOCHS=400
  export GIFNO_LEARNING_RATE=3e-4
  export GIFNO_BATCH_SIZE=8
  export GIFNO_NUM_WORKERS=4
  export GIFNO_CACHE_DATASET=true
  export GIFNO_USE_AMP=true
  export WANDB_PROJECT="${WANDB_PROJECT:-gifno_fdo_xt_loglo_pod}"
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-logpod_mix_lambda}"

  if [[ -f "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${SECRETS_FILE}"
    set +a
    export WANDB_MODE="${WANDB_MODE:-online}"
  else
    export WANDB_MODE="${WANDB_MODE:-offline}"
  fi

  mkdir -p "${GIFNO_MODEL_DIR}" "${GIFNO_RESULTS_DIR}"
  rm -f "${GIFNO_MODEL_DIR}/pod_modes.npy" "${GIFNO_MODEL_DIR}/pod_mean.npy" \
    "${GIFNO_MODEL_DIR}/pod_space.txt"
  cd "${SCRIPT_DIR}"
  python -u main.py
  echo "fine-tune wrote ${GIFNO_MODEL_DIR}/best_model.pt"
}

run_finetune() {
  # IID + dipping + three-layer. No basin OpenSees columns.
  check_data
  setup_venv
  activate_venv
  mkdir -p "${MIX}" "${MODEL_DIR}"
  if [[ ! -f "${MIX}/mix_tf_per_sample.npy" ]]; then
    echo "=== mix without basin columns ==="
    (
      export GIFNO_TF_DIR="${TF_DIR}"
      export GIFNO_H5_DIR="${H5_DIR}"
      cd "${SCRIPT_DIR}"
      python -u prepare_ood_mix_data.py \
        --box-root "${SCREEN}" \
        --out-dir "${MIX}" \
        --ood-scores-root "${PROJECT_ROOT}/checkpoints/ood_campaign_scores"
    )
  fi
  ln -sfn mix_tf_per_sample.npy "${MIX}/tf_per_sample.npy"
  ln -sfn mix_manifest.csv "${MIX}/manifest.csv"

  echo "=== log-POD mix fine-tune (no basin) ==="
  unset GIFNO_H5_DIR || true
  export GIFNO_DATA_ROOT="${SCREEN}"
  export GIFNO_TF_DIR="${MIX}"
  export GIFNO_MODEL_DIR="${MODEL_DIR}"
  export GIFNO_RESULTS_DIR="${MODEL_DIR}/results"
  export GIFNO_POD_SPACE=log
  export GIFNO_LOG_TF_LOSS=true
  export GIFNO_REINIT_POD_BRANCH=true
  export GIFNO_INIT_CHECKPOINT="${INIT_CKPT}"
  export GIFNO_LATENT_CHANNELS=128
  export GIFNO_POD_NUM_MODES=64
  export GIFNO_NUM_FNO_LAYERS=5
  export GIFNO_DEEPONET_LATENT_DIM=128
  export GIFNO_LOSS_RADIAL_WEIGHT=0.25
  export GIFNO_LOGLO_PATCH_SIZE=16,20
  export GIFNO_LOGLO_HF_NOISE_ALPHA=0.025
  export GIFNO_BAND_CURRICULUM=true
  export GIFNO_BAND_CURRICULUM_MODE=convergence
  export GIFNO_LOSS_BAND_BALANCED_WEIGHT=0.5
  export GIFNO_BAND_CURRICULUM_LR_RESTART=true
  export GIFNO_EARLY_STOP_PATIENCE=80
  export GIFNO_LR_SCHED_PATIENCE=30
  export GIFNO_NUM_EPOCHS=400
  export GIFNO_LEARNING_RATE=3e-4
  export GIFNO_BATCH_SIZE=8
  export GIFNO_NUM_WORKERS=4
  export GIFNO_CACHE_DATASET=true
  export GIFNO_USE_AMP=true
  export WANDB_PROJECT="${WANDB_PROJECT:-gifno_fdo_xt_loglo_pod}"
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-logpod_mix_nobasin}"
  export WANDB_MODE="${WANDB_MODE:-offline}"

  mkdir -p "${GIFNO_MODEL_DIR}" "${GIFNO_RESULTS_DIR}"
  rm -f "${GIFNO_MODEL_DIR}/pod_modes.npy" "${GIFNO_MODEL_DIR}/pod_mean.npy" \
    "${GIFNO_MODEL_DIR}/pod_space.txt"
  cd "${SCRIPT_DIR}"
  python -u main.py
  echo "fine-tune wrote ${GIFNO_MODEL_DIR}/best_model.pt"
}

launch_finetune() {
  if tmux has-session -t finetune 2>/dev/null; then
    echo "tmux session finetune is already running"
    exit 1
  fi
  local log="${PROJECT_ROOT}/checkpoints/logpod_mix_nobasin.log"
  mkdir -p "${PROJECT_ROOT}/checkpoints"
  tmux new-session -d -s finetune \
    "bash '${SCRIPT_DIR}/lambda_case7a.sh' finetune > '${log}' 2>&1"
  echo "started tmux session finetune"
  echo "log ${log}"
}

launch_tmux() {
  if tmux has-session -t case7a 2>/dev/null; then
    echo "tmux session case7a is already running"
    exit 1
  fi
  local log="${PROJECT_ROOT}/checkpoints/case7a_lambda.log"
  mkdir -p "${PROJECT_ROOT}/checkpoints"
  tmux new-session -d -s case7a \
    "bash '${SCRIPT_DIR}/lambda_case7a.sh' run > '${log}' 2>&1"
  echo "started tmux session case7a"
  echo "log ${log}"
}

cmd="${1:-launch}"
case "${cmd}" in
  setup) setup_venv ;;
  check) check_data ;;
  run) run_pipeline ;;
  launch) launch_tmux ;;
  finetune) run_finetune ;;
  launch-finetune) launch_finetune ;;
  *)
    echo "usage: bash lambda_case7a.sh [setup|check|run|launch|finetune|launch-finetune]" >&2
    exit 2
    ;;
esac
