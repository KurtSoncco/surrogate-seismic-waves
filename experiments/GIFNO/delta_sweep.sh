#!/bin/bash
# Submit 6 parallel GIFNO screening jobs (one hyperparameter change each).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO
#   source delta_env.sh
#   bash delta_sweep.sh              # round 1: 6 jobs, --limit 1000
#   bash delta_sweep.sh --limit 4000 # round 1 at 4000 samples
#
# Round 2 (latent_wide family, 9 jobs @ 2000 samples):
#   bash delta_sweep.sh --variants sweep_variants_r2.tsv --limit 2000
#
#   bash delta_sweep.sh --dry-run    # print what would be submitted
#
# After comparing in W&B, rerun the winner at full scale:
#   bash delta_sweep_rerun.sh lw_no_mine --variants sweep_variants_r2.tsv --full
#
# Round 3 (valley/linf fixes, 4 jobs @ 2000 samples):
#   bash delta_sweep.sh --variants sweep_variants_r3.tsv --limit 2000
#
# After round 3, full train with winner (default: lw_nm_logtf until W&B pick):
#   bash delta_sweep_rerun.sh lw_nm_logtf --variants sweep_variants_r3.tsv --full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=delta_env.sh
[[ -f "${SCRIPT_DIR}/delta_env.sh" ]] && source "${SCRIPT_DIR}/delta_env.sh"
# shellcheck source=delta_paths.sh
source "${SCRIPT_DIR}/delta_paths.sh"

DATA_ROOT="$(resolve_gifno_data_root)"
export GIFNO_DATA_ROOT="${DATA_ROOT}"
export GIFNO_TF_DIR="${DATA_ROOT}/transfer_function"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

exec python "${SCRIPT_DIR}/sweep_launch.py" "$@"
