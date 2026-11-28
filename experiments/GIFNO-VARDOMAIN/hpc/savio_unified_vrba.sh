#!/bin/bash
# Unified 0.1–10 Hz ablation ladder. One A5000, checkpoints under
# checkpoints/vardomain/v2_ablations/<ARM>. W&B runs offline; sync from a
# login node with `wandb sync` on the run directory.
#
#   ARM=A    sbatch experiments/GIFNO-VARDOMAIN/hpc/savio_unified_vrba.sh
#   ARM=B1   sbatch ...
#   ARM=B2   sbatch ...
#   ARM=C    sbatch ...
#
# Unified runs train on IID, the manifold, and the repeated-column slice.
# Dipping and three-layer are logged and held out of the loss, the POD, and
# every score that drives a checkpoint or the learning rate. Selection is the
# manifold validation split. Blind sections are reported only. The learning
# rate is a 60-epoch cosine and does not read validation.
# Arm A is the clean control (log-L1, PCA-16, 128-mode sqrt(f) POD).
# Arm B1 adds the hybrid relative, modal and LOGLO spectral loss.
# Arm B2 adds translation-equivariant local FiLM and vRBA.
# Arm C adds DeltaPhi retrieval on top of Arm B2.
# Cosine-bowl troughs are not in the cache the training set reads.
#
#SBATCH --job-name=vd_v2
#SBATCH --account=fc_tfsurrogate
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x_%j.out
#SBATCH --error=/global/scratch/users/kurtwal98/vardomain/logs/vd_%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/global/scratch/users/kurtwal98/vardomain/repo}"
VENV="${VENV:-/global/home/users/kurtwal98/surrogate-seismic-waves/.venv}"
VD_DIR="${PROJECT_ROOT}/experiments/GIFNO-VARDOMAIN"
CACHE="${CACHE:-/global/scratch/users/kurtwal98/vardomain/cache}"
ARM="${ARM:-A}"
CK_OUT="${CK_OUT:-/global/scratch/users/kurtwal98/vardomain/checkpoints/v2_ablations/${ARM}}"

module purge
module load python/3.11 2>/dev/null || true
module load cuda/12.6.0 2>/dev/null || module load cuda/12.4 2>/dev/null || true

cd "${PROJECT_ROOT}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
export WANDB_MODE=offline
mkdir -p "${CK_OUT}" "$(dirname "${CK_OUT}")"

EXTRA=()
case "${ARM}" in
  A) EXTRA+=(--loss log_l1) ;;
  B1) EXTRA+=(--loss hybrid_rel_log) ;;
  B2) EXTRA+=(--loss vrba_hybrid --vrba-potential quadratic --translation-equivariant) ;;
  C) EXTRA+=(--loss vrba_hybrid --vrba-potential quadratic --translation-equivariant --deltaphi) ;;
  *) echo "ARM must be A, B1, B2 or C, got ${ARM}" >&2; exit 1 ;;
esac

echo "=== unified arm ${ARM} -> ${CK_OUT} ==="
srun python -u "${VD_DIR}/train.py" \
  --cache "${CACHE}" \
  --out "${CK_OUT}" \
  --stage unified \
  --target residual \
  --seed 0 \
  --readout pod \
  --depth-pca \
  --encoding metric \
  --jitter \
  --whiten \
  --local-min-wavelength 10.0 \
  --pca-joint \
  --wandb \
  --wandb-project vardomain-v2 \
  --wandb-mode offline \
  "${EXTRA[@]}"
