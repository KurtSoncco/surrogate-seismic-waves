#!/bin/bash
#
# =====================================================================
# SLURM Directives for Savio3 A40 GPU - Configured for FCA
# =====================================================================

#SBATCH --job-name=FNO_TTF_Train         # Job name is descriptive
#SBATCH --account=kurtwal98         # Your project's Fine-Grained Allocation (FCA) account name
#SBATCH --partition=savio3_gpu           # The GPU partition
#SBATCH --nodes=1                        # Request 1 node
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1                       # Request 1 task (process)
#
# Processors per task:
# A40 GPUs in savio3_gpu require 8 CPUs per GPU
#SBATCH --cpus-per-task=8                
#
#Number and type of GPUs
#SBATCH --gres=gpu:a40:1                 # Request 1 A40 GPU

# Quality of Service (QoS) for A40 GPUs (Normal priority)
#SBATCH --qos=a40_gpu3_normal

# Wall clock limit:
#SBATCH --time=0:30:00                   # Set wall-clock limit (30 minutes)

# Output and Error logging
#SBATCH --output=%x.o%j             
#SBATCH --error=%x.e%j

# --- Email Notifications ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kurtwal98@berkeley.edu 

# =====================================================================
# Job Commands: Environment Setup and Execution
# =====================================================================

# Enable rigorous error checking
set -euo pipefail
trap 'echo "Error on line $LINENO" 1>&2' ERR

# --- 1. Define Project Root and Set CWD ---
# NOTE: Replace the placeholder below with the ABSOLUTE path to your 'surrogate-seismic-waves' directory
# Example: PROJECT_ROOT="/global/scratch/<username>/surrogate-seismic-waves"
PROJECT_ROOT="<PROJECT_ROOT_PATH_ON_SAVIO>" 
echo "Setting PROJECT_ROOT=$PROJECT_ROOT"

# Move to the project root so relative paths (like 'data/...' and '.venv/...' ) work
cd "$PROJECT_ROOT" || { echo "Cannot cd to $PROJECT_ROOT" 1>&2; exit 1; }

# Ensure python can find modules throughout the repo (e.g., 'wave_surrogate')
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# --- 2. Activate Virtual Environment (pyenv/uv) ---
ENV_PATH=".venv" 
VENV_ACTIVATE_SCRIPT="$PROJECT_ROOT/$ENV_PATH/bin/activate"

module purge

if [ -f "$VENV_ACTIVATE_SCRIPT" ]; then
    echo "Activating virtual environment: $VENV_ACTIVATE_SCRIPT"
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE_SCRIPT"
else
    echo "ERROR: Virtual environment activation script not found at $VENV_ACTIVATE_SCRIPT" 1>&2
    echo "Ensure you ran 'uv venv' and 'uv sync' in the $PROJECT_ROOT directory." 1>&2
    exit 1
fi

echo "Python interpreter: $(which python || echo 'python not found')"
nvidia-smi || echo "nvidia-smi failed - check GPU allocation" 1>&2

# --- 3. Execute the main script ---
# The main execution script is now targeted using its full path
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