# NCSA Delta GIFNO defaults (source from ~/.bashrc or delta_*.sh scripts).
#   source ~/surrogate-seismic-waves/experiments/GIFNO/delta_env.sh
#
# DELTA_ALLOC  = filesystem quota code (paths under /work/hdd/bgpu/...)
# DELTA_ACCOUNT = SLURM billing account (from `accounts`; use *-gpu for GPU jobs)
export DELTA_ALLOC="${DELTA_ALLOC:-bgpu}"
export DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgpu-delta-gpu}"
export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-/work/hdd/${DELTA_ALLOC}/${USER}/gifno_data}"

# W&B: do NOT put API keys in this file (it is tracked by git).
# Use experiments/GIFNO/lambda_secrets.env (gitignored) or export WANDB_API_KEY
# in your shell / Delta job env. delta_train.sh sources lambda_secrets.env if present.

# Silence wandb's noisy pydantic schema warnings in job logs (third-party, not
# actionable). Scoped to that module so our own warnings stay visible; our TF32
# deprecation is fixed in code, not hidden here. Override by pre-setting the var.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:::pydantic._internal._generate_schema}"
