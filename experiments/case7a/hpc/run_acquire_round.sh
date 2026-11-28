#!/bin/bash
# One AL round after the current ensemble and the last simulated sections exist.
#
#   MANIFESTS="sections/round0.csv sections/round1.csv" WINDOWS="$SCRATCH/windows_r0 $SCRATCH/windows_r1" \
#   MODEL_DIRS="$CK/pretrain_s0 $CK/pretrain_s1 $CK/pretrain_s2" ROUND=1 BATCH=12 \
#   bash hpc/run_acquire_round.sh
#
# Writes sections/round${ROUND}.csv. Submit those rows with stampede3_sections.slurm.
set -euo pipefail
CASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$CASE/../../.venv/bin/python}"
: "${ROUND:?}" "${MODEL_DIRS:?}" "${MANIFESTS:?}"
OUT="${OUT:-$CASE/sections/round${ROUND}.csv}"
flags=()
for m in $MODEL_DIRS; do flags+=(--model-dir "$m"); done
for m in $MANIFESTS; do flags+=(--manifest "$m"); done
for w in ${WINDOWS:-}; do flags+=(--windows "$w"); done
"$PY" "$CASE/acquire_sections.py" "${flags[@]}" --round "$ROUND" --batch "${BATCH:-12}" --out "$OUT"
echo "next: MANIFEST=$OUT sbatch $CASE/hpc/stampede3_sections.slurm"
