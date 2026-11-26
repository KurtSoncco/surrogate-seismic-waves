#!/bin/bash
# Record-length timing on Savio. Does not start the 8-master pilot and does
# not replace stampede3_pilot.slurm.
#
# One 2000 m master (p4m00, gradient, seed 0, manifest index 12) at 200 s
# and the same master at 400 s. Array task 0 is 200 s, task 1 is 400 s.
# Each task writes its own out-dir so the shared timing_ sample id cannot
# overwrite the other record. PAD_M stays 500 (SECTION_PAD_M unset).
#
# Account, partition, and qos match the finished OpenSees timings in
# savio_timing15.sh (savio4_htc, not a GPU partition). The home seiskit on
# Savio rejects xi_soil_base. SEISKIT_ROOT is the scratch copy that accepts
# it. The 5 Aug 2026 build is not used.
#
# A 15 s interior of 1500 m on this partition took about 17 s per step
# (nx 2500). This mesh is nx 3000 x 250. savio_normal caps a job at 3 days,
# which is shorter than a full 200 s or 400 s run at that rate. The header
# asks for the cap anyway.
#
# UmfPack on this wheel is umfpack_di (32-bit indices, no -useLongIndices).
# Gravity on the 3000 x 250 mesh dies in numeric factorization with
# "numeric analysis returns -1" (UMFPACK_ERROR_out_of_memory). Jobs
# 41507376_3 (80G), 41507396_3 (200G), and 41943643 (180G, both array tasks)
# all died in about 30 s at 5-7 GB RSS. The same UmfPack factors the 1500,
# 2000, and 2500 m padded meshes at this depth. Job 41946664 factored this
# same master with Mumps (gravity only, 95 s, 6.2 GB, two Newton iterations).
# Leave BASIN_SOLVER as Mumps. Record length, 500 m pads, and xi_soil_base
# stay as they are.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/pilot_record_timing/hpc
#   sbatch savio_record_timing.slurm
#
#SBATCH --job-name=p4_rectime
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/pilot_record_timing/logs/%A_%a.out
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/pilot_record_timing/logs/%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/global/scratch/users/kurtwal98/vardomain/pilot_record_timing}"
HPC_DIR="${HPC_DIR:-${ROOT}/hpc}"
CASE_DIR="${CASE_DIR:-${ROOT}/case7a}"
VENV="${VENV:-/global/home/users/kurtwal98/seiskit/.venv}"
# Scratch copy that accepts xi_soil_base. Do not point this at ~/seiskit.
export SEISKIT_ROOT="${SEISKIT_ROOT:-/global/scratch/users/kurtwal98/vardomain/timing15/seiskit}"
MANIFEST="${MANIFEST:-${HPC_DIR}/pilot_g4.csv}"
TIMING_CSV="${TIMING_CSV:-${HPC_DIR}/pilot_record_timing.csv}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# Mumps, not UmfPack. This OpenSees wheel's umfpack_di numeric phase returns
# -1 on the 3000 x 250 gravity system. See the header.
export BASIN_SOLVER="${BASIN_SOLVER:-Mumps}"
unset SECTION_PAD_M

if [[ ! -f "${CASE_DIR}/run_sections.py" ]]; then
  echo "missing ${CASE_DIR}/run_sections.py" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
WHEEL_LIB="$("${VENV}/bin/python" - <<'PY'
from pathlib import Path
import sys
lib = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "openseespylinux" / "lib"
print(lib)
PY
)"
export LD_LIBRARY_PATH="${WHEEL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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
        "Home seiskit on Savio rejects it. Use the timing15 scratch copy."
    )
print(f"seiskit accepts xi_soil_base: {root}", flush=True)
PY

TASK="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK}" ]]; then
  echo "Refusing a non-array submission. This file only times 200 s and 400 s." >&2
  exit 1
fi

read -r SEC INDEX < <(python - "${TIMING_CSV}" "${TASK}" <<'PY'
import csv
import sys
path, task = sys.argv[1:]
rows = list(csv.DictReader(open(path)))
seconds = sorted(int(float(r["timing_s"])) for r in rows)
if len(rows) != 2 or seconds != [200, 400]:
    raise SystemExit(f"{path} must time one master at 200 s and one at 400 s")
masters = {r["master_id"] for r in rows}
indexes = {r["manifest_index"] for r in rows}
if len(masters) != 1 or len(indexes) != 1:
    raise SystemExit("the 200 s and 400 s rows must be the same master")
task_i = int(task)
wanted = (200, 400)[task_i] if task_i in (0, 1) else None
if wanted is None:
    raise SystemExit(f"array task {task_i} is not the 200 s or 400 s timing")
row = next(r for r in rows if int(float(r["timing_s"])) == wanted)
print(wanted, int(float(row["manifest_index"])))
PY
)

OUT_DIR="${OUT_DIR:-${ROOT}/t${SEC}}"
mkdir -p "${OUT_DIR}/logs"
echo "timing ${SEC} s manifest_index=${INDEX} out=${OUT_DIR}; 8-master pilot not started"
python -u "${CASE_DIR}/run_sections.py" \
  --manifest "${MANIFEST}" \
  --index "${INDEX}" \
  --timing "${SEC}" \
  --out-dir "${OUT_DIR}"
echo "timing ${SEC} s finished. Record length is not chosen. The 8-master pilot was not started."
