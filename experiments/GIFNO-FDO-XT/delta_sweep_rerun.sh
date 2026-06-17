#!/bin/bash
# Rerun a single GIFNO-FDO-XT sweep variant.
#
#   bash delta_sweep_rerun.sh xt_p1_amsgrad_wide --limit 2000
#   bash delta_sweep_rerun.sh xt_p1_amsgrad_wide --full

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--variants FILE] [--limit N] [--full] [--dry-run]" >&2
    echo "Variants: xt_p1_amsgrad_wide xt_p1_amsgrad_wide_meters xt_p1_amsgrad_depth" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
