#!/usr/bin/env bash
# Copy the crop runner and meshes to Stampede3 and submit one SPR job.
# TACC still asks for a token, so this only works from a session that can ssh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="${STAMPEDE_HOST:-kurtsoncco1406@stampede3.tacc.utexas.edu}"
REMOTE="${STAMPEDE_REPO:-surrogate-seismic-waves}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
RSYNC=(rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20")

"${SSH[@]}" "$HOST" "mkdir -p ~/${REMOTE}/experiments/case7a/hpc ~/${REMOTE}/data/case7a/opensees_crops/meshes"
"${RSYNC[@]}" \
  "${ROOT}/experiments/case7a/case7_crops.py" \
  "${ROOT}/experiments/case7a/run_case7_crops.py" \
  "${HOST}:~/${REMOTE}/experiments/case7a/"
"${RSYNC[@]}" \
  "${ROOT}/experiments/case7a/hpc/stampede3_crops.slurm" \
  "${ROOT}/experiments/case7a/hpc/make_crop_tasks.py" \
  "${ROOT}/experiments/case7a/hpc/submit_crops.sh" \
  "${HOST}:~/${REMOTE}/experiments/case7a/hpc/"
"${RSYNC[@]}" \
  "${ROOT}/data/case7a/opensees_crops/meshes/" \
  "${HOST}:~/${REMOTE}/data/case7a/opensees_crops/meshes/"
"${SSH[@]}" "$HOST" "cd ~/${REMOTE}/experiments/case7a/hpc && sbatch stampede3_crops.slurm"
