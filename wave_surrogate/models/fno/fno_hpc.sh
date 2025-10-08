#!/bin/bash
# Minimal SLURM Directives (adjust time/memory as needed)
#SBATCH --job-name=FNO_MINIMAL
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0:30:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
set -euo pipefail

# Define the project root directly (Simplest path fix)
PROJECT_ROOT="/global/home/users/kurtwal98/surrogate-seismic-waves"

# Execute the job by sourcing the venv and running Python
srun /bin/bash -c "source $PROJECT_ROOT/.venv/bin/activate && \
                  cd $PROJECT_ROOT && \
                  export PYTHONPATH=$PROJECT_ROOT && \
                  python -u wave_surrogate/models/fno/main.py"