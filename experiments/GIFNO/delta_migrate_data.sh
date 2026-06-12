#!/bin/bash
# Move GIFNO data from home (/u/$USER) to bgpu work storage. Run on Delta login node.
#
#   cd ~/surrogate-seismic-waves
#   bash experiments/GIFNO/delta_migrate_data.sh
#
# Uses rsync then removes ~/gifno_data after verify (safe for large H5).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=delta_env.sh
source "${SCRIPT_DIR}/delta_env.sh"

SRC="${HOME}/gifno_data"
DEST="${GIFNO_DATA_ROOT}"

if [[ ! -d "${SRC}" ]]; then
    echo "Nothing to migrate: ${SRC} does not exist." >&2
    if [[ -d "${DEST}/h5" ]]; then
        echo "Data already at ${DEST}"
    fi
    exit 0
fi

if [[ -d "${DEST}" && -n "$(ls -A "${DEST}" 2>/dev/null)" ]]; then
    echo "ERROR: ${DEST} already has data. Set GIFNO_DATA_ROOT or remove it first." >&2
    exit 1
fi

echo "=== Migrate GIFNO data ==="
echo "  From: ${SRC}"
echo "  To:   ${DEST}"
echo "========================"

mkdir -p "$(dirname "${DEST}")"

echo "Syncing (may take a while for H5)..."
rsync -aP "${SRC}/" "${DEST}/"

echo ""
echo "--- Verify ---"
for req in h5 transfer_function/tf_per_sample.npy transfer_function/manifest.csv \
    transfer_function/recorder_x_idx.npy; do
    if [[ -e "${DEST}/${req}" ]]; then
        echo "  OK  ${DEST}/${req}"
    else
        echo "  MISSING  ${DEST}/${req}" >&2
        exit 1
    fi
done
echo "  H5 count: $(ls "${DEST}/h5"/run_*.h5 2>/dev/null | wc -l) files"

echo ""
read -r -p "Remove ${SRC}? [y/N] " ans
if [[ "${ans}" =~ ^[Yy]$ ]]; then
    rm -rf "${SRC}"
    echo "Removed ${SRC}. Home quota freed."
else
    echo "Kept ${SRC} — delete manually when satisfied: rm -rf ~/gifno_data"
fi

echo ""
echo "Add to ~/.bashrc:"
echo "  source ~/surrogate-seismic-waves/experiments/GIFNO/delta_env.sh"
echo ""
echo "Then: bash experiments/GIFNO/delta_setup.sh"
