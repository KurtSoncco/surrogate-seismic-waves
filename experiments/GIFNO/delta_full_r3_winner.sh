#!/bin/bash
# Full 7680-sample training for round-3 winner.
# Override variant after comparing W&B round-3 screen runs:
#   VARIANT=lw_nm_valley bash delta_full_r3_winner.sh
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO
#   source delta_env.sh
#   bash delta_full_r3_winner.sh
#   bash delta_full_r3_winner.sh --dry-run

set -euo pipefail

VARIANT="${VARIANT:-lw_nm_logtf}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep_rerun.sh" "${VARIANT}" \
    --variants sweep_variants_r3.tsv \
    --full \
    "$@"
