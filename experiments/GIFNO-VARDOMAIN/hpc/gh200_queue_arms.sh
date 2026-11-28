#!/bin/bash
# P64, B1, and B2 beside the live Arm A on the GH200.
# Each command is A's command with one arm's change and its own --out.
# This does not restart A and does not apply the later training protocol.
#
#   DRY=1 bash experiments/GIFNO-VARDOMAIN/hpc/gh200_queue_arms.sh
#   bash experiments/GIFNO-VARDOMAIN/hpc/gh200_queue_arms.sh

set -euo pipefail

cd /home/ubuntu/surrogate-seismic-waves
ROOT="checkpoints/vardomain/v2_ablations"
DRY="${DRY:-0}"

BASE=(
  python3 -u experiments/GIFNO-VARDOMAIN/train.py
  --cache data/vardomain/cache
  --stage unified
  --target residual
  --seed 0
  --readout pod
  --depth-pca
  --encoding metric
  --jitter
  --whiten
  --local-min-wavelength 10.0
  --pca-joint
  --wandb
  --wandb-project vardomain-v2
  --wandb-mode online
  --batch 16
  --workers 8
  --patience 15
  --lr-patience 13
)

launch() {
  local name="$1"
  shift
  local out="${ROOT}/${name}"
  if [[ -f "${out}/best_model.pt" ]]; then
    echo "refuse ${name}: ${out}/best_model.pt already exists" >&2
    exit 1
  fi
  local -a cmd=("${BASE[@]}" --out "${out}" "$@")
  printf '=== %s ===\n' "${name}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY}" == "1" ]]; then
    return
  fi
  mkdir -p "${out}"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "gpu_used_mib=${used} before ${name}"
  WANDB_NAME="${name}" WANDB_NOTES="${name} beside drawn-cloud-3, seed 0" \
    setsid nohup bash -c 'echo "start $(date -Is) $1"; shift; exec "$@"' _ "${name}" "${cmd[@]}" \
    > "${ROOT}/${name}.log" 2>&1 < /dev/null &
  echo $! > "${ROOT}/${name}.pid"
  echo "launched ${name} pid $(cat "${ROOT}/${name}.pid")"
  sleep 20
}

launch P64 --loss log_l1 --pca-rank 64
launch B1 --loss hybrid_rel_log
launch B2 --loss vrba_hybrid --vrba-potential quadratic --translation-equivariant
