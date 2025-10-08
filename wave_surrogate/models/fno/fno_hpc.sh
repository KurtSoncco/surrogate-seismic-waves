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

# =====================================================================
# Job Commands: Environment Setup and Execution
# =====================================================================

# Enable rigorous error checking
set -euo pipefail
trap 'echo "Error on line $LINENO" 1>&2' ERR

# --- 1. Define Project Root and Set CWD ---
PROJECT_ROOT="/global/home/users/kurtwal98/surrogate-seismic-waves" 
echo "Setting PROJECT_ROOT=$PROJECT_ROOT"

# Move to the project root so relative paths (like 'data/...' and '.venv/...' ) work
cd "$PROJECT_ROOT" || { echo "Cannot cd to $PROJECT_ROOT" 1>&2; exit 1; }


echo "Job finished at $(date)"