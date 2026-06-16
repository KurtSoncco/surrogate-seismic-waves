#!/bin/bash
# Full 7680-sample training for round-2 winner (lw_no_mine).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO
#   source delta_env.sh
#   bash delta_full_lw_no_mine.sh
#   bash delta_full_lw_no_mine.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep_rerun.sh" lw_no_mine \
    --variants sweep_variants_r2.tsv \
    --full \
    "$@"
