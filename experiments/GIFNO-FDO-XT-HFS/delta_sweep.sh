#!/bin/bash
# Submit GIFNO-FDO-XT-HFS sweep jobs.
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-HFS
#   source ../GIFNO/delta_env.sh
#   bash delta_sweep.sh --variants sweep_variants_hfs.tsv --limit 2000
#   bash delta_sweep_rerun.sh xt_wide_ref --full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIFNO_DIR="$(cd "${SCRIPT_DIR}/../GIFNO" && pwd)"
# shellcheck source=../GIFNO/delta_env.sh
[[ -f "${GIFNO_DIR}/delta_env.sh" ]] && source "${GIFNO_DIR}/delta_env.sh"
# shellcheck source=../GIFNO/delta_paths.sh
source "${GIFNO_DIR}/delta_paths.sh"

DATA_ROOT="$(resolve_gifno_data_root)"
export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

exec python "${SCRIPT_DIR}/sweep_launch.py" "$@"
