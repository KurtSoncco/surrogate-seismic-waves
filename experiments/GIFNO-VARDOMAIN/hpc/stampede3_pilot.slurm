#!/bin/bash
# Phase 4 pilot on Stampede3. Does not replace stampede3_manifold.slurm.
#
# Record length is not frozen. A bare submission times one 2000 m master
# (p4m00, gradient, seed 0, depth_m from the CSV, 500 m pads) at 200 s and
# at 400 s, then stops. It does not start the 8-master pilot.
#
# After those two wall times are known, choose RECORD_S and resubmit on a
# larger allocation. This file's default header is one node and will refuse
# the pilot even if RECORD_S is set:
#
#   PILOT_CONFIRMED=1 RECORD_S=<seconds> \
#     sbatch -N 8 -n 224 -t 24:00:00 stampede3_pilot.slurm
#
# Solver: experiments/case7a/run_sections.py
# Pads: PAD_M = 500. SECTION_PAD_M is left unset.
# Damping: AnalysisConfig receives xi_soil_base. SEISKIT_ROOT must be the
# package copied from ~/seiskit on 1 Nov 2026. The 5 Aug 2026 Stampede build
# ignored xi_soil_base; the check below exits when that parameter is absent.
# Mesh depth is the CSV depth_m (250), so the node packing follows the deep
# 3000 x 250 m job rather than the finished 150 m manifold job.
#
#   cd experiments/GIFNO-VARDOMAIN/hpc
#   sbatch stampede3_pilot.slurm
#
#SBATCH -J p4_pilot
#SBATCH -o p4_pilot_%j.o
#SBATCH -e p4_pilot_%j.e
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 12:00:00
#SBATCH -A TG-CIV260015

set -euo pipefail

HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VD_DIR="$(cd "${HPC_DIR}/.." && pwd)"
CASE_DIR="$(cd "${VD_DIR}/../case7a" && pwd)"
CASE_HPC="${CASE_DIR}/hpc"
PROJECT_ROOT="$(cd "${VD_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
# 1 Nov 2026 copy from ~/seiskit. Do not point this at the 5 Aug 2026 tree.
export SEISKIT_ROOT="${SEISKIT_ROOT:-/work2/09739/kurtsoncco1406/stampede3/seiskit}"
MANIFEST="$(realpath "${MANIFEST:-${HPC_DIR}/pilot_g4.csv}")"
TIMING_CSV="$(realpath "${TIMING_CSV:-${HPC_DIR}/pilot_record_timing.csv}")"
CORES_PER_TASK="${CORES_PER_TASK:-28}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 BASIN_SOLVER="${BASIN_SOLVER:-UmfPack}"
unset SECTION_PAD_M

module load python3 pylauncher 2>/dev/null || module load pylauncher
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

python - <<PY
import inspect
import os
import sys
root = os.environ["SEISKIT_ROOT"]
sys.path.insert(0, root)
from seiskit.config import AnalysisConfig
if "xi_soil_base" not in inspect.signature(AnalysisConfig.__init__).parameters:
    raise SystemExit(
        f"{root} does not accept xi_soil_base. "
        "That is the 5 Aug 2026 behavior. Use the 1 Nov 2026 copy from ~/seiskit."
    )
print(f"seiskit accepts xi_soil_base: {root}", flush=True)
PY

if [[ -z "${RECORD_S:-}" ]]; then
    OUT_DIR="${OUT_DIR:-${SCRATCH}/vardomain/pilot_g4_timing}"
    mkdir -p "${OUT_DIR}/logs"
    python - "${TIMING_CSV}" "${MANIFEST}" "${CASE_DIR}/run_sections.py" "${OUT_DIR}" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

timing_csv, manifest, runner, out_dir = sys.argv[1:]
rows = list(csv.DictReader(open(timing_csv)))
seconds = sorted(int(float(r["timing_s"])) for r in rows)
if len(rows) != 2 or seconds != [200, 400]:
    raise SystemExit(f"{timing_csv} must time one master at 200 s and one at 400 s")
masters = {r["master_id"] for r in rows}
indexes = {r["manifest_index"] for r in rows}
if len(masters) != 1 or len(indexes) != 1:
    raise SystemExit("the 200 s and 400 s rows must be the same master")
print(
    f"timing master={masters.pop()} index={indexes.pop()} at 200 s and 400 s; "
    "8-master pilot not started",
    flush=True,
)
for row in rows:
    sec = str(int(float(row["timing_s"])))
    log_path = Path(out_dir) / "logs" / f"timing_{sec}s.log"
    cmd = [
        sys.executable, "-u", runner,
        "--manifest", manifest,
        "--index", str(int(float(row["manifest_index"]))),
        "--timing", sec,
        "--out-dir", out_dir,
    ]
    print(" ".join(cmd), flush=True)
    with log_path.open("w") as log:
        subprocess.check_call(cmd, stdout=log, stderr=subprocess.STDOUT)
print("timing finished. Record length is not chosen. The 8-master pilot was not started.", flush=True)
PY
    exit 0
fi

if [[ "${PILOT_CONFIRMED:-0}" != "1" ]]; then
    echo "RECORD_S=${RECORD_S} but PILOT_CONFIRMED is not 1." >&2
    echo "Read the 200 s and 400 s timings, then resubmit with PILOT_CONFIRMED=1." >&2
    exit 1
fi
if [[ "${SLURM_NNODES:-1}" -lt 4 ]]; then
    echo "Refusing the 8-master run on ${SLURM_NNODES:-1} node(s)." >&2
    echo "Resubmit with -N 8 after RECORD_S is chosen from the timings." >&2
    exit 1
fi

OUT_DIR="${OUT_DIR:-${SCRATCH}/vardomain/pilot_g4}"
mkdir -p "${OUT_DIR}"
TASKS="${OUT_DIR}/tasks_${SLURM_JOB_ID:-local}.txt"
python "${CASE_HPC}/make_section_tasks.py" \
    --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" --duration "${RECORD_S}" > "${TASKS}"
echo "$(date -Is) tasks=$(wc -l < "${TASKS}") nodes=${SLURM_NNODES:-1} cores/task=${CORES_PER_TASK} record_s=${RECORD_S} out=${OUT_DIR}"
[[ -s "${TASKS}" ]] || { echo "nothing to run"; exit 0; }

cd "${OUT_DIR}"
python - "${TASKS}" "${CORES_PER_TASK}" <<'PY'
import sys
import pylauncher

pylauncher.ClassicLauncher(sys.argv[1], cores=int(sys.argv[2]), debug="job")
PY
echo "$(date -Is) done; failures: $(cat "${OUT_DIR}/failures.log" 2>/dev/null | wc -l)"
