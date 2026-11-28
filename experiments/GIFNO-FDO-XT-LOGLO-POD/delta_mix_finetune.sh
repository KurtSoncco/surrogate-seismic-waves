#!/bin/bash
# GPU job for run_mix_finetune.sh (NCSA Delta A100 by default).
#
#   cd ~/surrogate-seismic-waves/experiments/GIFNO-FDO-XT-LOGLO-POD
#   IID_DIR=... DIP_DIR=... TL_DIR=... SECTION_DIRS="..." DATA_OUT=... CK_OUT=... \
#   STAGE=pretrain SEEDS="0 1 2" sbatch --export=ALL delta_mix_finetune.sh
#
# Stampede3 H100 instead: sbatch -A ECS24003 -p h100 --gpus-per-node=1 \
#   --export=ALL delta_mix_finetune.sh (the module lines below are skipped there).
#
#SBATCH --job-name=loglo_mix
#SBATCH --account=bgpu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --output=loglo_mix.o%j
#SBATCH --error=loglo_mix.e%j

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
GIFNO_DIR="$(cd "${SCRIPT_DIR}/../GIFNO" && pwd)"
[[ -f "${GIFNO_DIR}/delta_env.sh" && "$(hostname)" == *delta* ]] && source "${GIFNO_DIR}/delta_env.sh"
if command -v module &>/dev/null && [[ "$(hostname)" == *delta* ]]; then
    module load gcc python 2>/dev/null || module load python
fi
export GIFNO_USE_AMP="${GIFNO_USE_AMP:-true}"
export GIFNO_NUM_WORKERS="${GIFNO_NUM_WORKERS:-6}"
export GIFNO_NUM_EPOCHS="${GIFNO_NUM_EPOCHS:-300}"
export GIFNO_WINDOW_SHIFT_MAX="${GIFNO_WINDOW_SHIFT_MAX:-6}"
export GIFNO_WINDOW_CONSISTENCY_WEIGHT="${GIFNO_WINDOW_CONSISTENCY_WEIGHT:-0.1}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a) STAGE=${STAGE:-all} SEEDS=${SEEDS:-0 1 2}"
bash "${SCRIPT_DIR}/run_mix_finetune.sh"
