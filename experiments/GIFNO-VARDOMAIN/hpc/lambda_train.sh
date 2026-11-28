#!/usr/bin/env bash
# Train the metric residual LOGLO on one Lambda A10 (24 GB). An A100 is not
# required. Checkpoints go to checkpoints/vardomain/.
#
#   ssh ubuntu@<LAMBDA_IP>
#   cd ~/surrogate-seismic-waves
#   bash experiments/GIFNO-VARDOMAIN/hpc/lambda_train.sh setup
#   bash experiments/GIFNO-VARDOMAIN/hpc/lambda_train.sh cache
#   bash experiments/GIFNO-VARDOMAIN/hpc/lambda_train.sh launch
#
# launch runs curriculum.py (stage 0 ablation, pair-weight pick, 3 seeds,
# stage 1 width jitter, stage 2 manifold replay) under tmux session `vardomain`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${VD_DIR}/../.." && pwd)"
VENV="${PROJECT_ROOT}/.venv"
DATA="${DATA:-${PROJECT_ROOT}/data/vardomain}"
CACHE="${CACHE:-${DATA}/cache}"
CK_OUT="${CK_OUT:-${PROJECT_ROOT}/checkpoints/vardomain}"
PYTHON="${PYTHON:-${VENV}/bin/python}"

# Fixed-window labels and the manifold interiors. Override if the copy lives elsewhere.
IID_H5="${IID_H5:-${PROJECT_ROOT}/data/gifno_screen/h5}"
DIP_H5="${DIP_H5:-${PROJECT_ROOT}/data/gifno_screen/ood_dipping/h5}"
TL_H5="${TL_H5:-${PROJECT_ROOT}/data/gifno_screen/ood_three_layer/h5}"
FIXED="${FIXED:-${DATA}/outcrop_fixed_window}"
MAN_H5="${MAN_H5:-${DATA}/manifold150/h5}"
MAN_LAB="${MAN_LAB:-${DATA}/manifold_labels/stride5}"
MAN_MANIFEST="${MAN_MANIFEST:-${DATA}/manifold150/manifest.csv}"

activate() { # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
}

setup() {
  [[ -x "${VENV}/bin/python" ]] || python3 -m venv --system-site-packages "${VENV}"
  activate
  python -m pip install -U pip
  python -m pip install 'numpy>=1.26,<2' scipy matplotlib pandas h5py hdf5plugin tqdm 'neuraloperator>=2.0.0'
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

cache() {
  activate
  mkdir -p "${CACHE}"
  "${PYTHON}" "${VD_DIR}/build_cache.py" \
    --fixed "iid=${IID_H5}:${FIXED}/iid:${IID_H5%/h5}/transfer_function/manifest.csv:${IID_H5%/h5}/transfer_function/recorder_x_idx.npy" \
    --fixed "ood_dipping=${DIP_H5}:${FIXED}/ood_dipping:${DIP_H5%/h5}/manifest.csv:${DIP_H5%/h5}/recorder_x_idx.npy" \
    --fixed "ood_three_layer=${TL_H5}:${FIXED}/ood_three_layer:${TL_H5%/h5}/manifest.csv:${TL_H5%/h5}/recorder_x_idx.npy" \
    --manifold-h5 "${MAN_H5}" --manifold-labels "${MAN_LAB}" --manifold-manifest "${MAN_MANIFEST}" \
    --out "${CACHE}"
}

run() {
  activate
  [[ -f "${CACHE}/freq.npy" ]] || cache
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  "${PYTHON}" -u "${VD_DIR}/curriculum.py" --cache "${CACHE}" --out "${CK_OUT}"
}

cmd="${1:-launch}"
case "$cmd" in
  setup) setup ;;
  cache) cache ;;
  run) run ;;
  launch)
    tmux has-session -t vardomain 2>/dev/null && { echo "tmux session vardomain already exists"; exit 1; }
    LOG="${LOG:-${PROJECT_ROOT}/train_vardomain.log}"
    tmux new-session -d -s vardomain "bash ${SCRIPT_DIR}/lambda_train.sh run 2>&1 | tee -a ${LOG}"
    echo "started tmux vardomain; tail -f ${LOG}"
    ;;
  *)
    echo "usage: $0 {setup|cache|run|launch}" >&2
    exit 1
    ;;
esac
