#!/usr/bin/env bash
# Hard stop 3 PM PDT (22:00 UTC): cut training at 2:15 PM so holdout + SVLAB finish.
set -euo pipefail
ROOT=/home/ubuntu/surrogate-seismic-waves
PY="${ROOT}/.venv/bin/python"
CK="${ROOT}/checkpoints/case7_generalization"
B="${CK}/outcrop_prior12/pretrain_s0"
CUTOFF=$(date -u -d '2026-10-30 21:15:00' +%s)

echo "[deadline] waiting until 21:15 UTC / 14:15 PDT  $(date -u)"
while [ "$(date -u +%s)" -lt "$CUTOFF" ]; do
  sleep 60
done

echo "[deadline] cutoff $(date -u)"
if pgrep -f "${ROOT}/.venv/bin/python -u main.py" >/dev/null; then
  echo "[deadline] stopping Bet B train so scoring can run"
  pkill -TERM -f "${ROOT}/.venv/bin/python -u main.py" || true
  sleep 20
  pkill -KILL -f "${ROOT}/.venv/bin/python -u main.py" || true
fi

if [[ -f "${B}/best_model.pt" && ! -f "${B}/holdout_pearson.json" ]]; then
  echo "[deadline] holdout Pearson Bet B"
  "$PY" "${ROOT}/experiments/GIFNO-FDO-XT-LOGLO-POD/score_holdout_pearson.py" \
    --model-dir "${B}" \
    --data "${ROOT}/data/gifno_screen/outcrop_tf/outcrop/mix_pretrain/holdout" \
    --out "${B}/holdout_pearson.json" \
    --arm pretrain_s0 || true
fi

if [[ ! -f "${CK}/residual/pretrain_s0/svlab/loglo_residual/metrics.json" ]]; then
  echo "[deadline] SVLAB residual"
  "$PY" "${ROOT}/experiments/case7a/score_svlab.py" --no-wipe \
    --out-dir "${CK}/residual/pretrain_s0/svlab" || true
fi
echo "[deadline] done $(date -u)"
