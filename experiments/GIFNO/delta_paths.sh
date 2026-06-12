#!/bin/bash
# NCSA Delta path helpers (source from other delta_*.sh scripts).
#
# Delta layout (not Savio/Berkeley):
#   Home:  /u/$USER          (~100 GB — code, secrets only)
#   Work:  /work/hdd/<alloc>/$USER   (job I/O / datasets; default alloc: bgpu)
#   $SCRATCH is set in batch jobs via `module reset`
#
# There is NO /scratch/$USER — that path will fail with "Permission denied".

# Default ACCESS allocation for this project (override: export DELTA_ALLOC=...)
: "${DELTA_ALLOC:=bgpu}"

delta_work_data_dir() {
    printf '/work/hdd/%s/%s/gifno_data\n' "${DELTA_ALLOC}" "${USER}"
}

resolve_gifno_data_root() {
    if [[ -n "${GIFNO_DATA_ROOT:-}" ]]; then
        printf '%s\n' "${GIFNO_DATA_ROOT}"
        return
    fi
    if [[ -n "${SCRATCH:-}" ]]; then
        printf '%s/gifno_data\n' "${SCRATCH}"
        return
    fi
    if [[ -n "${WORK:-}" ]]; then
        printf '%s/gifno_data\n' "${WORK}"
        return
    fi
    local work_dir
    work_dir="$(delta_work_data_dir)"
    if [[ -d "${work_dir}" ]]; then
        printf '%s\n' "${work_dir}"
        return
    fi
    if [[ -d "${HOME}/gifno_data" ]]; then
        printf '%s/gifno_data\n' "${HOME}"
        return
    fi
    printf '%s\n' "${work_dir}"
}

resolve_gifno_project_root() {
    if [[ -n "${PROJECT_ROOT:-}" ]]; then
        printf '%s\n' "${PROJECT_ROOT}"
        return
    fi
    if [[ -d "${HOME}/surrogate-seismic-waves" ]]; then
        printf '%s/surrogate-seismic-waves\n' "${HOME}"
        return
    fi
    printf '%s/surrogate-seismic-waves\n' "${HOME}"
}

# Default rsync/scp target on Delta (from local WSL).
resolve_delta_remote_data() {
    local delta_user="${1:-${DELTA_USER:-${USER}}}"
    if [[ -n "${REMOTE_DATA:-}" ]]; then
        printf '%s\n' "${REMOTE_DATA}"
        return
    fi
    printf '/work/hdd/%s/%s/gifno_data\n' "${DELTA_ALLOC}" "${delta_user}"
}
