#!/bin/bash
# Rerun a single GIFNO-FDO sweep variant.
#
#   bash delta_sweep_rerun.sh fdn_h1 --limit 2000
#   bash delta_sweep_rerun.sh fdn_h1 --full

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--variants FILE] [--limit N] [--full] [--dry-run]" >&2
    echo "Arch: fdn_h1 fdn_h1_wide fdn_h1_depth" >&2
    echo "Combo: lw_nm_h1_valley (sweep_variants_combo.tsv)" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
