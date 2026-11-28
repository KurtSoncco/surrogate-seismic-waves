#!/usr/bin/env bash
# Submit the 64 Sobol 2000 m sections on Stampede3 (account ECS24003).
#
#   ssh stampede3.tacc.utexas.edu
#   bash experiments/case7a/hpc/submit_round0.sh
set -euo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-${HPC_DIR}/../sections/round0.csv}"
NODES="${NODES:-2}"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not on PATH. On Stampede3 run:" >&2
  echo "  MANIFEST=$MANIFEST sbatch -N $NODES ${HPC_DIR}/stampede3_sections.slurm" >&2
  exit 1
fi
cd "$HPC_DIR"
exec sbatch -N "$NODES" --export=ALL,MANIFEST="$MANIFEST" stampede3_sections.slurm
