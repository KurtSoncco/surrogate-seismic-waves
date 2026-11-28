#!/usr/bin/env bash
# Train residual and absolute outcrop LOGLO on the same mix (plan dual-head).
#
# Residual: TF / TF_1D, 1D prior as input. Absolute: TF_2D directly, 1D still an input.
# Needs SECTION_DIRS (cropped Stampede3 windows). Same env as run_mix_finetune.sh.
#
#   DATA_OUT=$WORK/outcrop_tf CK_OUT=$WORK/checkpoints \
#   IID_DIR=... DIP_DIR=... TL_DIR=... CORNER_DIR=... SECTION_DIRS=".../windows_round0" \
#   TARGETS="outcrop residual" bash run_dual_head.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS="${TARGETS:-outcrop residual}"
: "${DATA_OUT:?set DATA_OUT}" "${CK_OUT:?set CK_OUT}"
for target in $TARGETS; do
  echo "=== dual-head TF_TARGET=$target ==="
  TF_TARGET="$target" DATA_OUT="${DATA_OUT}/${target}" CK_OUT="${CK_OUT}/${target}" \
    bash "$SCRIPT_DIR/run_mix_finetune.sh"
done
