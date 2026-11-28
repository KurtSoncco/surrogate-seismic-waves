#!/bin/bash
# Corrected from-scratch learning curve. Do not use gh200_learning_curve.sh:
# its fixed-basis path used to load the full-data network weights.
# Run only after the feasibility gate in analysis/preregistration/decision.json opens.
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE="${CACHE:?set CACHE to the vardomain cache}"
for FOLD in 0 1 2 3 4; do
  for FRAC in 0.25 0.50 1.00; do
    python train.py \
      --cache "$CACHE" --stage unified --protocol audit \
      --cv-fold "$FOLD" --cv-folds 5 --cv-seed 0 \
      --subset-seed 0 --model-seed 0 --sampler-seed 0 \
      --iid-group-frac "$FRAC" --fixed-batches 1560 \
      --epochs 60 --patience 1000 --skip-test \
      --out "../../checkpoints/vardomain/audit/iid${FRAC}_f${FOLD}"
  done
done
