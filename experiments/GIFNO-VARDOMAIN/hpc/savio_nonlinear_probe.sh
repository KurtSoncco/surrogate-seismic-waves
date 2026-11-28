#!/bin/bash
# Eight-epoch stage-0 probe: log-POD control against the depth autoencoder
# and the nonlinear spectral decoder. One A5000. Writes probe_gate.json and
# exits non-zero when the gate fails, so a dependent stage job does not start.
# Leaves early_width and the live curriculum directories alone.
#
#   ssh savio
#   cd /global/scratch/users/kurtwal98/vardomain/repo
#   sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_nonlinear_probe.sh
#
#SBATCH --job-name=vd_nl_probe
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=08:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.o%j
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x.e%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
OUT="${OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/nonlinear_probe}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

[[ -f "${CACHE}/freq.npy" ]] || { echo "missing ${CACHE}/freq.npy" >&2; exit 1; }
mkdir -p "${OUT}"

echo "=== nonlinear probe -> ${OUT} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

COMMON=(
  --cache "${CACHE}" --encoding metric
  --stage 0 --target residual --pair-weight 0 --seed 0
  --epochs 8 --patience 8
)

srun python -u "${VD_DIR}/train.py" "${COMMON[@]}" --out "${OUT}/pod" --overwrite

srun python -u "${VD_DIR}/train.py" "${COMMON[@]}" \
  --depth-ae --readout nonlinear --recon-weight 0.1 \
  --local-min-wavelength 0 --hfp-kernel 4 --hfp-stride 4 \
  --out "${OUT}/nonlinear" --overwrite

srun python -u - "${OUT}" << 'PY'
import csv, json, math, sys
from pathlib import Path

out = Path(sys.argv[1])
agree_floor = 0.0136

def column(path, key):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    vals = []
    for row in rows:
        val = float(row[key])
        if not math.isfinite(val):
            return None
        vals.append(val)
    return vals

pod_val = column(out / "pod" / "history.csv", "val_score")
nl_val = column(out / "nonlinear" / "history.csv", "val_score")
nl_train = column(out / "nonlinear" / "history.csv", "train_log_l1")
nl_mem = column(out / "nonlinear" / "history.csv", "peak_mem_gb")
with (out / "nonlinear" / "history.csv").open() as fh:
    nl_agree = [float(row["agreement"]) for row in csv.DictReader(fh)]

reasons = []
if not pod_val or not nl_val:
    reasons.append("a validation score was missing or non-finite")
if not nl_train or not (nl_train[-1] < nl_train[0]):
    reasons.append("nonlinear train_log_l1 did not fall")
if pod_val and nl_val and min(nl_val) > 2.0 * min(pod_val):
    reasons.append("best nonlinear val_score is more than twice the POD control")
if nl_mem and max(nl_mem) >= 20.0:
    reasons.append("nonlinear peak memory reached 20 GB")
if not any(math.isfinite(a) and a < agree_floor for a in nl_agree):
    reasons.append("no nonlinear epoch met the agreement floor")

gate = {
    "passed": not reasons,
    "reasons": reasons,
    "pod_best_val": None if not pod_val else min(pod_val),
    "nonlinear_best_val": None if not nl_val else min(nl_val),
    "nonlinear_train_log_l1": nl_train,
    "nonlinear_peak_mem_gb": None if not nl_mem else max(nl_mem),
    "nonlinear_min_agreement": None if not nl_agree else min(nl_agree),
}
(out / "probe_gate.json").write_text(json.dumps(gate, indent=2))
print(json.dumps(gate, indent=2), flush=True)
if reasons:
    raise SystemExit(1)
PY

echo "probe gate passed ${OUT}/probe_gate.json"
