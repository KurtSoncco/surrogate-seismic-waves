#!/bin/bash
# Pretrain / fine-tune residual LOGLO on Stampede3 H100 (account ECS24003).
#
#   IID_DIR=... DIP_DIR=... TL_DIR=... SECTION_DIRS="..." \
#   DATA_OUT=$SCRATCH/outcrop_tf CK_OUT=$SCRATCH/checkpoints STAGE=pretrain \
#   SEEDS="0 1 2" sbatch --export=ALL stampede3_mix_finetune.slurm
#
#SBATCH -J loglo_mix
#SBATCH -o loglo_mix_%j.o
#SBATCH -e loglo_mix_%j.e
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A ECS24003

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
module load python3 cuda 2>/dev/null || true
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
export PYTHON="${PYTHON:-${VENV_PATH}/bin/python}"
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-true}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-8}"
export GIFNO_NUM_EPOCHS="${GIFNO_NUM_EPOCHS:-300}"
export GIFNO_WINDOW_SHIFT_MAX="${GIFNO_WINDOW_SHIFT_MAX:-6}"
export GIFNO_WINDOW_CONSISTENCY_WEIGHT="${GIFNO_WINDOW_CONSISTENCY_WEIGHT:-0.1}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a) STAGE=${STAGE:-all}"
bash "${SCRIPT_DIR}/run_mix_finetune.sh"
