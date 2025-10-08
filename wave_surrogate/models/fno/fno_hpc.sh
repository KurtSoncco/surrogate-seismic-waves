#!/bin/bash
#
# =====================================================================
# SLURM Directives for Savio3 A40 GPU
# =====================================================================

#SBATCH --job-name=FNO_TTF_Train    # A descriptive name for your job
#SBATCH --account=fc_tfsurrogate    # Your project's Fine-Grained Allocation (FCA) account name
#SBATCH --qos=a40_gpu3_normal       # Quality of Service (QoS) for A40 GPUs with normal priority
#SBATCH --partition=savio3_gpu      # The GPU partition

# --- Resource Allocation ---
# Request 1 A40 GPU
# The --gres flag should specify the type of GPU in savio3_gpu for FCA jobs
#SBATCH --gres=gpu:a40:1            

# Request 1 Node (The partition uses per-core allocation, but nodes=1 is standard)
#SBATCH --nodes=1 

# Request 1 Task (PyTorch is single-task per GPU unless using DDP/torchrun)
#SBATCH --ntasks=1                  

# CPU-to-GPU Ratio Requirement: A40 GPUs in savio3_gpu require 8 CPUs per GPU
#SBATCH --cpus-per-task=8           

# Set wall-clock limit (Default savio_normal max is 72 hours)
# Adjust this based on expected runtime.
#SBATCH --time=0:30:00             

# Output and Error logging (Recommended for tracking)
# Output will be written to a file named FNO_TTF_Train.o<JOB_ID>
#SBATCH --output=%x.o%j             
#SBATCH --error=%x.e%j

# --- Email Notifications (Optional) ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kurtwal98@berkeley.edu 

# =====================================================================
# Job Commands: Environment Setup and Execution
# =====================================================================

set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

# --- Resolve script location and repository root (robust) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# script lives in .../wave_surrogate/models/fno -> climb up 3 to reach repo root
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
echo "Using PROJECT_ROOT=$PROJECT_ROOT"

# Move to the project root so relative paths inside the project behave reliably
cd "$PROJECT_ROOT" || { echo "Cannot cd to $PROJECT_ROOT"; exit 1; }

# Ensure python can import modules from the repo
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Configurable environment/module settings (override by exporting before sbatch)
CONDA_ENV="${CONDA_ENV:-my_fno_env}"
MODULE_NAME="${MODULE_NAME:-anaconda3}"

# Load modules and activate conda if available
module purge
module load "$MODULE_NAME" || echo "module load $MODULE_NAME returned non-zero (continuing)"

if command -v conda >/dev/null 2>&1; then
  # make 'conda activate' available in non-interactive shells
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    # Fallback to legacy source activate (may or may not work)
    source activate "$CONDA_ENV" || true
  fi
else
  echo "conda not found in PATH; please ensure your environment is available on the node"
fi

echo "Python interpreter: $(which python || echo 'python not found')"
echo "CUDA available (nvidia-smi output):"
nvidia-smi || true

# Determine which main script to run (can override MAIN_PY env var)
MAIN_PY=""$PROJECT_ROOT/wave_surrogate/models/fno/main.py""
if [ ! -f "$MAIN_PY" ]; then
  echo "Error: Main script $MAIN_PY not found!"
  exit 1
fi

echo "Executing main script: $MAIN_PY"
# Pass any sbatch arguments through to the python script
srun python -u "$MAIN_PY" "$@"

# Deactivate conda if possible
if command -v conda >/dev/null 2>&1; then
  conda deactivate || true
fi

echo "Job finished at $(date)"