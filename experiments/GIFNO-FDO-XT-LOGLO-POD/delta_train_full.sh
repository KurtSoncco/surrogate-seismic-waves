#!/bin/bash
# GIFNO-FDO-XT-LOGLO-POD full-dataset training (7680 samples, no --limit).
# Submitted automatically by: bash delta_sweep.sh --full --name <variant>
#
#   sbatch delta_train_full.sh

#SBATCH --job-name=loglo_pod_full
#SBATCH --account=bgpu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --output=loglo_pod_full.o%j
#SBATCH --error=loglo_pod_full.e%j

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

exec bash "${SCRIPT_DIR}/delta_train.sh"
