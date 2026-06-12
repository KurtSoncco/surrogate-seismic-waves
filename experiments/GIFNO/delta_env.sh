# NCSA Delta GIFNO defaults (source from ~/.bashrc or delta_*.sh scripts).
#   source ~/surrogate-seismic-waves/experiments/GIFNO/delta_env.sh
#
# DELTA_ALLOC  = filesystem quota code (paths under /work/hdd/bgpu/...)
# DELTA_ACCOUNT = SLURM billing account (from `accounts`; use *-gpu for GPU jobs)
export DELTA_ALLOC="${DELTA_ALLOC:-bgpu}"
export DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgpu-delta-gpu}"
export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-/work/hdd/${DELTA_ALLOC}/${USER}/gifno_data}"
