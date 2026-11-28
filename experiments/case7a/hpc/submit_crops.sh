#!/usr/bin/env bash
# Submit the five Case 7 crops on Stampede3 (account ECS24003, one SPR node).
#
#   bash experiments/case7a/hpc/submit_crops.sh
set -euo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not on PATH. On Stampede3 run:" >&2
  echo "  cd ${HPC_DIR} && sbatch stampede3_crops.slurm" >&2
  exit 1
fi
cd "$HPC_DIR"
exec sbatch stampede3_crops.slurm
