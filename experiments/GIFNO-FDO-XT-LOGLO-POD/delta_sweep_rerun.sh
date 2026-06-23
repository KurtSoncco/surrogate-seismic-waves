#!/bin/bash
# Rerun a single LOGLO-POD sweep variant (screen or full dataset).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD
#   source ../GIFNO/delta_env.sh
#
#   bash delta_sweep_rerun.sh loglo_pod_ref --limit 500
#   bash delta_sweep_rerun.sh loglo_pod_ref --full
#   bash delta_sweep_rerun.sh loglo_pod_ref --full --dry-run

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--variants FILE] [--limit N] [--full] [--dry-run]" >&2
    echo "Examples: loglo_pod_ref loglo_pod_bs16 loglo_pod_pod64" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
