#!/usr/bin/env bash
# Stage a 2k–3k residual screen pack from Box for the cloud agent or a local copy.
#
# Run on Kurt-Asus (Box mounted). Does not upload; writes a folder you can tar
# and share via a Box download link.
#
#   bash experiments/DeepONet-Residual/stage_screen_pack.sh
#   bash experiments/DeepONet-Residual/stage_screen_pack.sh --n 2000 --out ~/gifno_screen
#
# Then:
#   tar -C "$OUT" -cvf ~/gifno_screen.tar .
#   # upload ~/gifno_screen.tar to Box → Share → paste URL in the agent chat

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
N=3000
SEED=42
BOX="${GIFNO_DATA_ROOT:-/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data}"
if [[ ! -d "${BOX}/h5" && -d "/mnt/box_lab/Projects/Neural Operator/data/h5" ]]; then
    BOX="/mnt/box_lab/Projects/Neural Operator/data"
fi
OUT="${HOME}/gifno_screen"

usage() {
    echo "Usage: $0 [--n 2000|3000] [--seed 42] [--box PATH] [--out PATH]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n) N="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --box) BOX="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

CACHE_TAG="n${N}_seed${SEED}"
H5_SRC="${BOX}/h5"
TF_SRC="${BOX}/transfer_function"

if [[ ! -d "${H5_SRC}" || ! -f "${TF_SRC}/manifest.csv" ]]; then
    echo "ERROR: Box data not found at ${BOX}" >&2
    echo "Set GIFNO_DATA_ROOT or --box to the folder that contains h5/ and transfer_function/" >&2
    exit 1
fi

echo "Box:  ${BOX}"
echo "Out:  ${OUT}"
echo "Tag:  ${CACHE_TAG}"
mkdir -p "${OUT}/h5" "${OUT}/transfer_function" "${OUT}/ood_dipping" "${OUT}/ood_three_layer"

echo "=== copy transfer_function (full TF cache) ==="
rsync -a --include='*.npy' --include='*.csv' --exclude='models/**' --exclude='results/**' \
    "${TF_SRC}/" "${OUT}/transfer_function/"

echo "=== stratified indices ${CACHE_TAG} ==="
export GIFNO_DATA_ROOT="${BOX}"
export GIFNO_H5_DIR="${H5_SRC}"
export GIFNO_TF_DIR="${TF_SRC}"
H5_LIST="$(mktemp)"
(
    cd "${REPO_ROOT}"
    uv run python experiments/DeepONet-Residual/select_indices.py \
        --cache-tag "${CACHE_TAG}" --print-h5 > "${H5_LIST}"
)
mapfile -t H5_NAMES < <(grep -E '\\.h5$' "${H5_LIST}" || true)
rm -f "${H5_LIST}"
echo "Copying ${#H5_NAMES[@]} H5 files..."

copied=0
missing=0
for name in "${H5_NAMES[@]}"; do
    src="${H5_SRC}/${name}"
    if [[ ! -f "${src}" ]]; then
        echo "  MISSING ${src}" >&2
        missing=$((missing + 1))
        continue
    fi
    cp -n "${src}" "${OUT}/h5/${name}" || true
    copied=$((copied + 1))
done
echo "H5 copied=${copied} missing=${missing}"

echo "=== copy OOD corpora ==="
for name in ood_dipping ood_three_layer; do
    if [[ -d "${BOX}/${name}" ]]; then
        rsync -a "${BOX}/${name}/" "${OUT}/${name}/"
        echo "  OK ${name} ($(du -sh "${OUT}/${name}" | awk '{print $1}'))"
    else
        echo "  SKIP missing ${BOX}/${name}" >&2
    fi
done

{
    echo "cache_tag=${CACHE_TAG}"
    echo "n=${N}"
    echo "seed=${SEED}"
    echo "source=${BOX}"
    echo "h5_copied=${copied}"
    echo "h5_missing=${missing}"
} > "${OUT}/SCREEN_README.txt"

echo ""
echo "Pack ready at ${OUT}"
echo "  tar -C \"${OUT}\" -cvf ~/gifno_screen.tar ."
echo "  export GIFNO_DATA_ROOT=\"${OUT}\""
echo "  export GIFNO_H5_DIR=\"${OUT}/h5\""
echo "  export GIFNO_TF_DIR=\"${OUT}/transfer_function\""
echo "  export GIFNO_OOD_DIPPING=\"${OUT}/ood_dipping\""
echo "  export GIFNO_OOD_THREE_LAYER=\"${OUT}/ood_three_layer\""
