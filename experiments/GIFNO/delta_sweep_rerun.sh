#!/bin/bash
# Rerun a single sweep variant (e.g. winner after screening).
#
#   bash delta_sweep_rerun.sh latent_wide --limit 1000
#   bash delta_sweep_rerun.sh lw_no_mine_fno --variants sweep_variants_r2.tsv --full

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_name> [--variants FILE] [--limit N] [--full] [--dry-run]" >&2
    echo "Round 1: baseline h1_strong fno_wide latent_wide freq_loss no_mining" >&2
    echo "Round 2: lw_anchor lw_no_mine lw_fno lw_no_mine_fno ... (see sweep_variants_r2.tsv)" >&2
    exit 1
fi

VARIANT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/delta_sweep.sh" --name "${VARIANT}" "$@"
