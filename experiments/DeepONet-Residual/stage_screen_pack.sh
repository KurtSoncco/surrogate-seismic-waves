#!/usr/bin/env bash
# Stage a GIFNO screen pack without the full 7680-sample h5/ tree.
#
# Layout written to DEST (default: ./data/gifno_screen):
#   transfer_function/   # copy all (tf_per_sample.npy, freq, manifest, recorder_x)
#   h5/                  # only stratified run_*.h5 for n=3000 (unless --ood-only)
#   ood_dipping/
#   ood_three_layer/
#   sample_indices.npy
#
# Usage:
#   experiments/DeepONet-Residual/stage_screen_pack.sh [DEST]
#   N_SAMPLES=2000 ./experiments/DeepONet-Residual/stage_screen_pack.sh /tmp/gifno_screen
#   OOD_ONLY=1 ./experiments/DeepONet-Residual/stage_screen_pack.sh
#
# Then: export GIFNO_DATA_ROOT=/path/to/gifno_screen
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${GIFNO_DATA_ROOT:-/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data}"
DEST="${1:-${ROOT}/data/gifno_screen}"
N_SAMPLES="${N_SAMPLES:-3000}"
SEED="${SEED:-42}"
OOD_ONLY="${OOD_ONLY:-0}"
CACHE_TAG="n${N_SAMPLES}_seed${SEED}"

copy_tree() {
  local from="$1" to="$2"
  mkdir -p "$(dirname "$to")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$from" "$to"
  else
    cp -a "$from" "$to"
  fi
}

echo "[stage] SRC=${SRC}"
echo "[stage] DEST=${DEST}"
mkdir -p "$DEST"

if [[ ! -d "$SRC/ood_dipping" || ! -d "$SRC/ood_three_layer" ]]; then
  echo "[stage] WARN: OOD folders missing under SRC. Expected ood_dipping and ood_three_layer." >&2
fi

if [[ -d "$SRC/ood_dipping" ]]; then
  echo "[stage] copying ood_dipping"
  rm -rf "$DEST/ood_dipping"
  copy_tree "$SRC/ood_dipping" "$DEST/ood_dipping"
fi
if [[ -d "$SRC/ood_three_layer" ]]; then
  echo "[stage] copying ood_three_layer"
  rm -rf "$DEST/ood_three_layer"
  copy_tree "$SRC/ood_three_layer" "$DEST/ood_three_layer"
fi

if [[ "$OOD_ONLY" == "1" ]]; then
  echo "[stage] OOD_ONLY=1 — skipped IID transfer_function/ and h5/"
  echo "[stage] done → ${DEST}"
  exit 0
fi

if [[ ! -d "$SRC/transfer_function" ]]; then
  echo "[stage] ERROR: ${SRC}/transfer_function missing (need tf_per_sample.npy)" >&2
  exit 1
fi
echo "[stage] copying transfer_function"
rm -rf "$DEST/transfer_function"
copy_tree "$SRC/transfer_function" "$DEST/transfer_function"

echo "[stage] writing stratified ${CACHE_TAG} indices (no Residual RF screen)"
cd "$ROOT"
uv run python experiments/DeepONet-Residual/residual_signed.py \
  --cache-tag "$CACHE_TAG" --indices-only

IDX="${ROOT}/experiments/DeepONet-Residual/cache/${CACHE_TAG}/sample_indices.npy"
if [[ ! -f "$IDX" ]]; then
  echo "[stage] ERROR: expected indices at ${IDX}" >&2
  exit 1
fi
cp -a "$IDX" "$DEST/sample_indices.npy"

echo "[stage] copying stratified H5 files (n=${N_SAMPLES})"
mkdir -p "$DEST/h5"
uv run python - << PY
import csv
from pathlib import Path
import numpy as np

src = Path(${SRC@Q}) / "h5"
dst = Path(${DEST@Q}) / "h5"
idx = np.load(Path(${IDX@Q}))
man_path = Path(${DEST@Q}) / "transfer_function" / "manifest.csv"
with open(man_path, newline="") as f:
    rows = list(csv.DictReader(f))
copied = 0
missing = 0
for i in idx:
    i = int(i)
    name = Path(rows[i]["h5_path"]).name
    src_f = src / name
    if not src_f.is_file():
        missing += 1
        continue
    dst_f = dst / name
    if not dst_f.exists():
        dst_f.write_bytes(src_f.read_bytes())
    copied += 1
print(f"[stage] h5 copied={copied} missing={missing}", flush=True)
if missing:
    raise SystemExit(f"missing {missing} H5 files under {src}")
PY

echo "[stage] done → ${DEST}"
echo "[stage] export GIFNO_DATA_ROOT=${DEST}"
