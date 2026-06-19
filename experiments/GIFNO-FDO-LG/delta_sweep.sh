#!/bin/bash
# Submit GIFNO-FDO-LG sweep jobs.
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-LG
#   source ../GIFNO/delta_env.sh
#   bash delta_sweep.sh --variants sweep_variants_lg.tsv --limit 2000
#   bash delta_sweep_rerun.sh lg_phase1_ref --limit 2000
#
# Phase 1: PRETRAIN_CHECKPOINT defaults to xt_lat128_d128 (override if needed).
# Phase 2/3: export PRETRAIN_CHECKPOINT (phase1 LG ckpt) and
#            XT_ANCHOR_CHECKPOINT (XT ckpt) before submitting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIFNO_DIR="$(cd "${SCRIPT_DIR}/../GIFNO" && pwd)"
# shellcheck source=../GIFNO/delta_env.sh
[[ -f "${GIFNO_DIR}/delta_env.sh" ]] && source "${GIFNO_DIR}/delta_env.sh"
# shellcheck source=../GIFNO/delta_paths.sh
source "${GIFNO_DIR}/delta_paths.sh"

DATA_ROOT="$(resolve_gifno_data_root)"
export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_H5_DIR="${DATA_ROOT}/h5"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"

# Default XT init for phase-1 transfer when not already set.
if [[ -z "${PRETRAIN_CHECKPOINT:-}" ]]; then
    for cand in \
        "${GIFNO_TF_DIR}/models/fdo_xt/sweep/n2000/xt_lat128_d128/best_model.pt" \
        "${GIFNO_TF_DIR}/models/fdo_xt/sweep/xt_lat128_d128/best_model.pt" \
        "${GIFNO_TF_DIR}/models/fdo_xt/best_model.pt"; do
        if [[ -f "${cand}" ]]; then
            export PRETRAIN_CHECKPOINT="${cand}"
            echo "PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT}"
            break
        fi
    done
fi

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

exec python "${SCRIPT_DIR}/sweep_launch.py" "$@"
