#!/usr/bin/env bash
# Generate 32 deep-section rows and submit them on Stampede3.
#
#   ssh stampede3.tacc.utexas.edu
#   bash experiments/GIFNO-VARDOMAIN/hpc/submit_deep.sh
set -euo pipefail
VD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-${VD_DIR}/hpc/deep250.csv}"
NODES="${NODES:-8}"
if [[ ! -f "${MANIFEST}" ]]; then
  python "${VD_DIR}/generate_deep.py" --n "${N:-32}" --out "${MANIFEST}"
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "On Stampede3 run:" >&2
  echo "  MANIFEST=${MANIFEST} OUT_DIR=\$SCRATCH/vardomain/deep250_pad sbatch -N ${NODES} ${VD_DIR}/hpc/stampede3_deep.slurm" >&2
  exit 1
fi
cd "${VD_DIR}/hpc"
exec sbatch -N "${NODES}" --export=ALL,MANIFEST="${MANIFEST}",OUT_DIR="${OUT_DIR:-${SCRATCH}/vardomain/deep250_pad}" stampede3_deep.slurm
