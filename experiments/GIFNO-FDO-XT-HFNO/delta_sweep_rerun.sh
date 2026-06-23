#!/bin/bash
# Rerun a single HFNO sweep variant (screen or full dataset).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-HFNO
#   source ../GIFNO/delta_env.sh
#
#   bash delta_sweep_rerun.sh hfno_lat96_d64 --limit 2000
#   bash delta_sweep_rerun.sh hfno_ref --full
#   bash delta_sweep_rerun.sh hfno_lat96_d64 --full --dry-run

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--variants FILE] [--limit N] [--full] [--dry-run]" >&2
    echo "Examples: hfno_ref hfno_lat96_d64 hfno_lat96_d128 hfno_depth" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
