#!/bin/bash
# Day-2 arms on the P64 base. Same protocol as drawn-cloud-3 and P64.
# Log-L1 stays the loss. The hybrid loss is not used. DeltaPhi is not started.
#
#   DRY=1 bash experiments/GIFNO-VARDOMAIN/hpc/gh200_queue_day2.sh
#   bash experiments/GIFNO-VARDOMAIN/hpc/gh200_queue_day2.sh

set -euo pipefail

cd /home/ubuntu/surrogate-seismic-waves
ROOT="checkpoints/vardomain/v2_ablations"
DRY="${DRY:-0}"

BASE=(
  python3 -u experiments/GIFNO-VARDOMAIN/train.py
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
  local -a cmd=("${BASE[@]}" "$@")
  # A later --seed, --loss, --pca-rank, or --out replaces the base value.
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
  WANDB_NAME="${name}" WANDB_NOTES="${name} on the P64 base, log-L1, seed from the command" \
    setsid nohup bash -c 'echo "start $(date -Is) $1"; shift; exec "$@"' _ "${name}" "${cmd[@]}" \
    > "${ROOT}/${name}.log" 2>&1 < /dev/null &
  echo $! > "${ROOT}/${name}.pid"
  echo "launched ${name} pid $(cat "${ROOT}/${name}.pid")"
  sleep 20
}

launch P64film --out "${ROOT}/P64film" --seed 0 --pca-rank 64 --loss vrba_log_l1 --vrba-potential quadratic --translation-equivariant
launch P128 --out "${ROOT}/P128" --seed 0 --pca-rank 128 --loss log_l1
launch P64s1 --out "${ROOT}/P64s1" --seed 1 --pca-rank 64 --loss log_l1
