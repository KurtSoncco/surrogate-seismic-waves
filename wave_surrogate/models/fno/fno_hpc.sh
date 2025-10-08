#!/bin/bash
#SBATCH --job-name=test 
#SBATCH --account=fc_tfsurrogate
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=8
#
#Number and type of GPUs
#SBATCH --gres=gpu:a40:1

#SBATCH --qos=a40_gpu3_normal

# Wall clock limit:
#SBATCH --time=00:00:30
## Command(s) to run (example):
./a.out