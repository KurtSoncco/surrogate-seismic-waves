# NCSA Delta GIFNO defaults (source from ~/.bashrc or delta_*.sh scripts).
#   source ~/surrogate-seismic-waves/experiments/GIFNO/delta_env.sh
export DELTA_ALLOC="${DELTA_ALLOC:-bgpu}"
export GIFNO_DATA_ROOT="${GIFNO_DATA_ROOT:-/work/hdd/${DELTA_ALLOC}/${USER}/gifno_data}"
