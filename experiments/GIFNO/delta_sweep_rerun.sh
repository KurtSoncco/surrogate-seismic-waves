#!/bin/bash
# Rerun a single sweep variant (e.g. winner after screening).
#
#   bash delta_sweep_rerun.sh fno_wide           # screen again (--limit 1000)
#   bash delta_sweep_rerun.sh fno_wide --full  # full 7680-sample training

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--full] [--dry-run]" >&2
    echo "Variants: baseline h1_strong fno_wide latent_wide freq_loss no_mining" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
