#!/usr/bin/env bash
# Absolute-input, log-POD LOGLO on outcrop labels (Case 7 generalization plan).
#
#   OUTCROP_TF_DIR=../../data/gifno_screen/outcrop_tf/residual_iid \
#   TF_TARGET=residual MODEL_DIR=../../checkpoints/loglo_abs_residual_s0 \
#   bash run_abs_train.sh [--limit N]
#
# OUTCROP_TF_DIR comes from build_outcrop_dataset.py; its manifest holds
# absolute H5 paths, so GIFNO_H5_DIR is left unset (campaigns share basenames).
# TF_TARGET must match the --target the dataset was built with. Everything the
# scorer needs to rebuild the model is written to MODEL_DIR/run_env.json.
# Extra GIFNO_* variables in the environment are passed through and recorded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${OUTCROP_TF_DIR:?set OUTCROP_TF_DIR to a build_outcrop_dataset.py output}"
: "${MODEL_DIR:?set MODEL_DIR}"
export TF_TARGET="${TF_TARGET:-residual}"

built="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['target'])" "$OUTCROP_TF_DIR/dataset.json")"
if [[ "$built" != "$TF_TARGET" ]]; then
  echo "TF_TARGET=$TF_TARGET but $OUTCROP_TF_DIR was built as $built" >&2
  exit 1
fi

unset GIFNO_H5_DIR
export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-$(dirname "$OUTCROP_TF_DIR")}"
export GIFNO_TF_DIR="$OUTCROP_TF_DIR"
export GIFNO_MODEL_DIR="$MODEL_DIR"
export GIFNO_RESULTS_DIR="${GIFNO_RESULTS_DIR:-$MODEL_DIR/results}"

export GIFNO_INPUT_ENCODING=absolute
export GIFNO_POD_LAYOUT="${GIFNO_POD_LAYOUT:-shared}"
export GIFNO_NZ_MAX="${GIFNO_NZ_MAX:-160}"
export GIFNO_DIMNORM_FILM="${GIFNO_DIMNORM_FILM:-true}"
export GIFNO_PRIOR_CHANNELS="${GIFNO_PRIOR_CHANNELS:-$([[ $TF_TARGET == residual ]] && echo 12 || echo 0)}"
export GIFNO_POD_SPACE=log
# Residual targets are TF/TF_1D ~ 1, whose log is ~0: a relative norm of the
# log is then ill-conditioned, so score the ratio itself (~ relative log error).
export GIFNO_LOG_TF_LOSS="${GIFNO_LOG_TF_LOSS:-$([[ $TF_TARGET == residual ]] && echo false || echo true)}"
export GIFNO_POD_NUM_MODES="${GIFNO_POD_NUM_MODES:-64}"
export GIFNO_LATENT_CHANNELS="${GIFNO_LATENT_CHANNELS:-128}"
export GIFNO_NUM_FNO_LAYERS="${GIFNO_NUM_FNO_LAYERS:-5}"
export GIFNO_DEEPONET_LATENT_DIM="${GIFNO_DEEPONET_LATENT_DIM:-128}"
export GIFNO_LOSS_RADIAL_WEIGHT="${GIFNO_LOSS_RADIAL_WEIGHT:-0.25}"
export GIFNO_FREQ_BAND_HIGH="${GIFNO_FREQ_BAND_HIGH:-2.0,10.0}"
export GIFNO_EARLY_STOP_PATIENCE="${GIFNO_EARLY_STOP_PATIENCE:-140}"
export GIFNO_LR_SCHED_PATIENCE="${GIFNO_LR_SCHED_PATIENCE:-40}"
export GIFNO_SELECTION_METRIC="${GIFNO_SELECTION_METRIC:-pearson_trusted}"
export GIFNO_BATCH_SIZE="${GIFNO_BATCH_SIZE:-8}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-4}"
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-true}"
export GIFNO_SEED="${GIFNO_SEED:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-gifno_case7_generalization}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename "$MODEL_DIR")}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$GIFNO_MODEL_DIR" "$GIFNO_RESULTS_DIR"
python3 - "$GIFNO_MODEL_DIR/run_env.json" <<'EOF'
import json, os, sys
keep = {k: v for k, v in os.environ.items() if k.startswith("GIFNO_") and k not in ("GIFNO_MODEL_DIR", "GIFNO_RESULTS_DIR")}
keep["TF_TARGET"] = os.environ["TF_TARGET"]
json.dump(dict(sorted(keep.items())), open(sys.argv[1], "w"), indent=2)
EOF

PY="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
echo "=== absolute-input log-POD LOGLO ($TF_TARGET) ==="
echo "TF_DIR=$GIFNO_TF_DIR  MODEL_DIR=$GIFNO_MODEL_DIR  NZ_MAX=$GIFNO_NZ_MAX  PRIOR_CHANNELS=$GIFNO_PRIOR_CHANNELS"
cd "$SCRIPT_DIR"
exec "$PY" -u main.py "$@"
