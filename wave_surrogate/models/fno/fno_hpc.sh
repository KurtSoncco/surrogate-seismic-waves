#!/bin/bash
#
# =====================================================================
# SLURM Directives for Savio3 A40 GPU - Using pyenv/uv environment
# =====================================================================

#SBATCH --job-name=FNO_TTF_Train    # A descriptive name for your job
#SBATCH --account=fc_tfsurrogate    # Your project's Fine-Grained Allocation (FCA) account name
#SBATCH --qos=a40_gpu3_normal       # QoS for A40 GPUs (Normal priority)
#SBATCH --partition=savio3_gpu      # The GPU partition

# --- Resource Allocation ---
# Request 1 A40 GPU
#SBATCH --gres=gpu:a40:1            
# Request 1 Node
#SBATCH --nodes=1 
# Request 1 Task
#SBATCH --ntasks=1                  
# CPU-to-GPU Ratio Requirement: A40 GPUs require 8 CPUs per GPU on Savio3
#SBATCH --cpus-per-task=8           
# Wall-clock limit (30 minutes)
#SBATCH --time=0:30:00             
# Output and Error logging
#SBATCH --output=%x.o%j             
#SBATCH --error=%x.e%j

# --- Email Notifications ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kurtwal98@berkeley.edu 

# =====================================================================
# Job Commands: Environment Setup and Execution with CWD fix
# =====================================================================

# Enable rigorous error checking
set -euo pipefail
trap 'echo "Error on line $LINENO" 1>&2' ERR

# --- 1. Resolve script location and repository root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FIX: The submission script (and main.py) is in /.../fno/. 
# The PROJECT_ROOT (surrogate-seismic-waves) is 2 levels up.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
echo "Using determined PROJECT_ROOT=$PROJECT_ROOT"

# Move to the project root so relative paths (like 'data/...' and '.venv/...' ) work
cd "$PROJECT_ROOT" || { echo "Cannot cd to $PROJECT_ROOT" 1>&2; exit 1; }

# Ensure python can find modules throughout the repo (e.g., 'wave_surrogate')
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# --- 2. Activate Virtual Environment (pyenv/uv) ---
ENV_PATH="${VENV_PATH:-.venv}" 
MODULE_NAME="${MODULE_NAME:-python}" 

module purge
# If you need to load a specific system-wide module for Python/CUDA, do it here
# module load "$MODULE_NAME" 

# Activate the virtual environment created by 'uv venv'
VENV_ACTIVATE_SCRIPT="$PROJECT_ROOT/$ENV_PATH/bin/activate"

if [ -f "$VENV_ACTIVATE_SCRIPT" ]; then
    echo "Activating virtual environment: $VENV_ACTIVATE_SCRIPT"
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE_SCRIPT"
else
    echo "ERROR: Virtual environment activation script not found at $VENV_ACTIVATE_SCRIPT" 1>&2
    echo "Make sure you ran 'uv venv' and 'uv sync' in the $PROJECT_ROOT directory." 1>&2
    exit 1
fi

echo "Python interpreter: $(which python || echo 'python not found')"
echo "CUDA available (nvidia-smi output):"
nvidia-smi || true

# --- 3. Execute the main script ---
# The main script's absolute path is known: PROJECT_ROOT/wave_surrogate/models/fno/main.py
MAIN_PY="$PROJECT_ROOT/wave_surrogate/models/fno/main.py"
if [ ! -f "$MAIN_PY" ]; then
    echo "ERROR: Main script not found at the expected path: $MAIN_PY" 1>&2
    exit 1
fi

echo "Executing main script: $MAIN_PY"
# -u is for unbuffered output
srun python -u "$MAIN_PY" "$@"

# --- 4. Cleanup ---
# Deactivate the virtual environment
deactivate || true

echo "Job finished at $(date)"