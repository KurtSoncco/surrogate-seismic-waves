#!/bin/bash
# IID learning curve scored by manifold grouped cross-validation.
# Waits until P128s1 and P128c have exited. The winner is chosen on the
# selection score alone (mean of IID and manifold log-L1). Dipping and
# three-layer are written down and do not decide the base.
#
# Each arm trains five folds. A fold holds out part of the 22 paired section
# groups, trains for 60 epochs, and records the holdout every epoch.
# The 100% arm is trained under that same protocol. After the curve, four
# seeds of neck 128 with contact features are trained on the full split.
#
#   bash experiments/GIFNO-VARDOMAIN/hpc/gh200_learning_curve.sh

set -euo pipefail

cd /home/ubuntu/surrogate-seismic-waves
ROOT="checkpoints/vardomain/v2_ablations"
PY="${PY:-python3}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "${NUMBA_CACHE_DIR}"

live() {
  pgrep -af 'experiments/GIFNO-VARDOMAIN/train.py' | grep -E 'P128s1|P128c' | grep -v 'pgrep' || true
}

if [[ -n "${CURVE_WINNER:-}" ]]; then
  echo "curve starts now from ${CURVE_WINNER}, without waiting for a running training $(date -Is)"
else
echo "waiting for P128s1 and P128c to exit"
while [[ -n "$(live)" ]]; do
  date -Is
  live
  sleep 120
done
echo "both trainings have exited $(date -Is)"
fi

score_one() {
  local name="$1"
  if [[ -n "${CURVE_WINNER:-}" ]]; then
    return
  fi
  if [[ -f "${ROOT}/${name}_score.json" ]]; then
    echo "keep ${name}_score.json"
    return
  fi
  "${PY}" -u experiments/GIFNO-VARDOMAIN/score_readouts.py "${ROOT}/${name}" data/vardomain/cache cuda \
    > "${ROOT}/${name}_score.json"
  echo "scored ${name}"
}

if [[ -z "${CURVE_WINNER:-}" ]]; then
score_one P128
score_one P128s1
score_one P128c

"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("checkpoints/vardomain/v2_ablations")
def load(name):
    return json.loads((root / f"{name}_score.json").read_text())
p128, s1, c = load("P128"), load("P128s1"), load("P128c")
sel = "selection_0p1_10"
readouts = ("dip_0p1_10", "three_0p1_10")
delta = abs(p128[sel] - s1[sel])
bar = min(p128[sel], s1[sel]) - 2.0 * delta
clears = c[sel] < bar
winner = "P128c" if clears else "P128n128"
decision = {
    "rule": "selection score only; dipping and three-layer are readouts",
    "p128": {"selection": p128[sel], "dip": p128["dip_0p1_10"], "three": p128["three_0p1_10"], "epoch": p128.get("epoch")},
    "p128s1": {"selection": s1[sel], "dip": s1["dip_0p1_10"], "three": s1["three_0p1_10"], "epoch": s1.get("epoch")},
    "p128c": {"selection": c[sel], "dip": c["dip_0p1_10"], "three": c["three_0p1_10"], "epoch": c.get("epoch")},
    "delta_selection": delta,
    "bar_selection": bar,
    "clears_selection": clears,
    "readouts_not_used": list(readouts),
    "winner": winner,
    "cv_epochs": 60,
    "cv_folds": 5,
}
(root / "learning_curve_decision.json").write_text(json.dumps(decision, indent=2))
print(json.dumps(decision, indent=2))
PY
fi

if [[ -n "${CURVE_WINNER:-}" ]]; then
  WINNER="${CURVE_WINNER}"
else
  WINNER="$(${PY} -c 'import json; print(json.load(open("checkpoints/vardomain/v2_ablations/learning_curve_decision.json"))["winner"])')"
fi
echo "winner ${WINNER}"

BASE=(
  "${PY}" -u experiments/GIFNO-VARDOMAIN/train.py
  --cache data/vardomain/cache
  --stage unified
  --target residual
  --readout pod
  --depth-pca
  --encoding metric
  --jitter
  --whiten
  --local-min-wavelength 10.0
  --pca-joint
  --pca-rank 128
  --protocol p128
  --fixed-batches 1560
  --fixed-basis-dir "${ROOT}/${WINNER}"
  --epochs 60
  --patience 1000
  --wandb
  --wandb-project vardomain-v2
  --wandb-mode online
  --batch 16
  --workers 16
  --fast-val
  --lr-patience 13
  --seed 0
  --cv-folds 5
  --cv-seed 0
)
if [[ "${WINNER}" == "P128c" ]]; then
  BASE+=(--narrow 32 --contact-features)
else
  BASE+=(--narrow 128)
fi

fold_done() {
  local out="$1"
  [[ -f "${out}/cv_epochs.jsonl" ]] || return 1
  grep -q '"epoch": 59' "${out}/cv_epochs.jsonl"
}

launch() {
  local name="$1"
  shift
  local out="${ROOT}/${name}"
  if fold_done "${out}"; then
    echo "skip ${name}: epoch 59 already scored"
    return
  fi
  mkdir -p "${out}"
  local -a cmd=("${BASE[@]}" --out "${out}" "$@")
  if [[ -f "${out}/best_model.pt" ]]; then
    cmd+=(--overwrite)
  fi
  printf '=== %s ===\n' "${name}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  WANDB_NAME="${name}" WANDB_NOTES="${name} CV learning curve on ${WINNER}" \
    setsid nohup bash -c 'echo "start $(date -Is) $1"; shift; exec "$@"' _ "${name}" "${cmd[@]}" \
    > "${ROOT}/${name}.log" 2>&1 < /dev/null &
  echo $! > "${ROOT}/${name}.pid"
  echo "launched ${name} pid $(cat "${ROOT}/${name}.pid")"
  while kill -0 "$(cat "${ROOT}/${name}.pid")" 2>/dev/null; do
    sleep 60
  done
  echo "finished ${name} $(date -Is)"
}

for fold in 0 1 2 3 4; do
  launch "lc_g25_f${fold}" --iid-group-frac 0.25 --cv-fold "${fold}"
  launch "lc_g50_f${fold}" --iid-group-frac 0.5 --cv-fold "${fold}"
  launch "lc_s8_f${fold}" --iid-seeds-per-group 8 --cv-fold "${fold}"
  launch "lc_full_f${fold}" --cv-fold "${fold}"
  launch "lc_man50_f${fold}" --manifold-group-frac 0.5 --cv-fold "${fold}"
done

"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("checkpoints/vardomain/v2_ablations")
arms = ("lc_g25", "lc_g50", "lc_s8", "lc_full", "lc_man50")
epochs = (40, 59)
summary = {}
for arm in arms:
    summary[arm] = {}
    for epoch in epochs:
        rows = []
        for fold in range(5):
            path = root / f"{arm}_f{fold}" / "cv_epochs.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if int(row["epoch"]) == epoch:
                    rows.append(row)
        if not rows:
            continue
        def mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return None if not vals else sum(vals) / len(vals)
        summary[arm][f"epoch_{epoch + 1}"] = {
            "n_folds": len(rows),
            "selection": mean("selection"),
            "iid": mean("iid"),
            "manifold_holdout": mean("manifold"),
            "manifold_lateral": mean("lateral"),
        }
(root / "learning_curve_cv_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "cross-validation summary written $(date -Is)"

# Four seeds of the final configuration: neck 128 plus contact features.
# Full split, own basis, the same early-stopping recipe as P128n128.
FINAL=(
  "${PY}" -u experiments/GIFNO-VARDOMAIN/train.py
  --cache data/vardomain/cache
  --stage unified
  --target residual
  --readout pod
  --depth-pca
  --encoding metric
  --jitter
  --whiten
  --local-min-wavelength 10.0
  --pca-joint
  --pca-rank 128
  --narrow 128
  --contact-features
  --protocol p128
  --loss log_l1
  --wandb
  --wandb-project vardomain-v2
  --wandb-mode online
  --batch 16
  --workers 16
  --fast-val
  --patience 15
  --lr-patience 13
)
for seed in 0 1 2 3; do
  name="neck128c_s${seed}"
  out="${ROOT}/${name}"
  if [[ -f "${out}/best_model.pt" ]]; then
    echo "skip ${name}: best_model.pt exists"
    continue
  fi
  mkdir -p "${out}"
  cmd=("${FINAL[@]}" --out "${out}" --seed "${seed}")
  printf '=== %s ===\n' "${name}"
  WANDB_NAME="${name}" WANDB_NOTES="${name} neck 128 plus contact features" \
    setsid nohup bash -c 'echo "start $(date -Is) $1"; shift; exec "$@"' _ "${name}" "${cmd[@]}" \
    > "${ROOT}/${name}.log" 2>&1 < /dev/null &
  echo $! > "${out}.pid"
  while kill -0 "$(cat "${out}.pid")" 2>/dev/null; do
    sleep 60
  done
  echo "finished ${name} $(date -Is)"
done
echo "learning curve and neck128c seeds finished $(date -Is)"
