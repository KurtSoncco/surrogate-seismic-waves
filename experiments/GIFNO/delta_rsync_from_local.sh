#!/bin/bash
# Upload GIFNO data from local WSL (Box) to NCSA Delta scratch.
#
# Prerequisites:
#   go-lab   # mount Box at /mnt/box_lab
#   ssh ksonccosinchi@login.delta.ncsa.illinois.edu   # Duo MFA once
#
# Usage:
#   bash experiments/GIFNO/delta_rsync_from_local.sh
#   bash experiments/GIFNO/delta_rsync_from_local.sh --h5-only
#   bash experiments/GIFNO/delta_rsync_from_local.sh --tf-only

set -euo pipefail

DELTA_USER="${DELTA_USER:-ksonccosinchi}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
REMOTE_DATA="${REMOTE_DATA:-/scratch/${DELTA_USER}/gifno_data}"
LOCAL_DATA="${LOCAL_DATA:-/mnt/box_lab/Projects/Neural Operator/data}"

RSYNC_OPTS=(-avzP --partial)

sync_tf() {
    echo "=== rsync transfer_function (~250 MB) ==="
    rsync "${RSYNC_OPTS[@]}" \
        "${LOCAL_DATA}/transfer_function/" \
        "${DELTA_USER}@${DELTA_HOST}:${REMOTE_DATA}/transfer_function/"
}

sync_h5() {
    echo "=== rsync h5 (3120 files — may take hours) ==="
    rsync "${RSYNC_OPTS[@]}" \
        "${LOCAL_DATA}/h5/" \
        "${DELTA_USER}@${DELTA_HOST}:${REMOTE_DATA}/h5/"
}

if [[ ! -d "${LOCAL_DATA}/h5" ]]; then
    echo "ERROR: Box data not found at ${LOCAL_DATA}" >&2
    echo "Run: go-lab   (mounts Box at /mnt/box_lab)" >&2
    exit 1
fi

echo "Local:  ${LOCAL_DATA}"
echo "Remote: ${DELTA_USER}@${DELTA_HOST}:${REMOTE_DATA}"
echo ""

ssh "${DELTA_USER}@${DELTA_HOST}" "mkdir -p ${REMOTE_DATA}/h5 ${REMOTE_DATA}/transfer_function"

case "${1:-all}" in
    --tf-only) sync_tf ;;
    --h5-only) sync_h5 ;;
    all|--all|"")
        sync_tf
        sync_h5
        ;;
    *)
        echo "Usage: $0 [--tf-only|--h5-only]" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Remote verify ==="
ssh "${DELTA_USER}@${DELTA_HOST}" "
    echo 'H5:        \$(ls ${REMOTE_DATA}/h5/run_*.h5 2>/dev/null | wc -l) files'
    ls -lh ${REMOTE_DATA}/transfer_function/*.npy ${REMOTE_DATA}/transfer_function/*.csv 2>/dev/null
"

echo "Done."
