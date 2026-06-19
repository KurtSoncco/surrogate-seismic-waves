#!/bin/bash
# Rerun a single GIFNO-FDO-LG sweep variant.
#
#   bash delta_sweep_rerun.sh lg_phase1_ref --limit 2000
#   bash delta_sweep_rerun.sh lg_phase2_ref --limit 2000

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--variants FILE] [--limit N] [--full] [--dry-run]" >&2
    echo "Variants: lg_phase1_ref lg_phase1_gated lg_phase2_ref lg_phase3_ref" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
