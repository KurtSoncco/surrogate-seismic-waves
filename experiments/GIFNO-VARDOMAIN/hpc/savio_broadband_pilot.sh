#!/bin/bash
# Write the paired-width 0.05–50 Hz pilot job list and run the local numerical
# checks (graded mesh, Ormsby band flatness, homogeneous outcrop |TF| = 1).
# The OpenSees solves themselves are the rows of jobs.csv. This script does
# not submit those solves.
#
#SBATCH --job-name=vd_bbpilot
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/broadband_pilot}"

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -u experiments/GIFNO-VARDOMAIN/broadband_pilot.py --out "${OUT}" --geologies 8
