#!/bin/bash
# One-time environment setup on NCSA Delta (run on login node).
#
#   ssh ksonccosinchi@login.delta.ncsa.illinois.edu
#   git clone https://github.com/KurtSoncco/surrogate-seismic-waves.git
#   cd surrogate-seismic-waves && bash experiments/GIFNO/delta_setup.sh
#
# After setup, copy W&B secrets from local (use ~ — $SCRATCH expands locally on WSL):
#   scp experiments/GIFNO/lambda_secrets.env \
#     ksonccosinchi@login.delta.ncsa.illinois.edu:~/surrogate-seismic-waves/experiments/GIFNO/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=delta_env.sh
[[ -f "${SCRIPT_DIR}/delta_env.sh" ]] && source "${SCRIPT_DIR}/delta_env.sh"
# shellcheck source=delta_paths.sh
source "${SCRIPT_DIR}/delta_paths.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"
DATA_ROOT="$(resolve_gifno_data_root)"

echo "=== Delta GIFNO setup ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SCRATCH=${SCRATCH:-unset}"
echo "WORK=${WORK:-unset}"
echo "DELTA_ALLOC=${DELTA_ALLOC:-unset}"
echo "========================="

# Delta RH9: keep default Cray PE + cudatoolkit; Savio-style python/3.11 is unavailable.
load_delta_python() {
    if module is-loaded python 2>/dev/null; then
        return 0
    fi
    module load gcc python 2>/dev/null || module load python
}
load_delta_python
DELTA_PYTHON="$(which python)"
echo "Python: ${DELTA_PYTHON} ($("${DELTA_PYTHON}" --version))"

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${PROJECT_ROOT}"
uv venv .venv --python "${DELTA_PYTHON}"
source .venv/bin/activate
uv sync

echo ""
echo "--- W&B check ---"
SECRETS_FILE="${GIFNO_DIR}/lambda_secrets.env"
if [[ -f "${SECRETS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    set +a
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        WANDB_API_KEY="${WANDB_API_KEY//$'\r'/}"
        export WANDB_API_KEY
    fi
fi
if WANDB_API_KEY="${WANDB_API_KEY:-}" wandb whoami 2>/dev/null; then
    echo "W&B: logged in (via WANDB_API_KEY or prior login)"
else
    echo "W&B: NOT logged in — scp lambda_secrets.env from local or run: wandb login"
fi

echo ""
echo "--- Data check ---"
for req in \
    "${DATA_ROOT}/h5" \
    "${DATA_ROOT}/transfer_function/tf_per_sample.npy" \
    "${DATA_ROOT}/transfer_function/manifest.csv" \
    "${DATA_ROOT}/transfer_function/recorder_x_idx.npy"; do
    if [[ -e "${req}" ]]; then
        echo "  OK  ${req}"
    else
        echo "  MISSING  ${req}"
    fi
done

if [[ -d "${DATA_ROOT}/h5" ]]; then
    echo "  H5 count: $(ls "${DATA_ROOT}/h5"/run_*.h5 2>/dev/null | wc -l)"
fi

echo ""
echo "--- SLURM accounts (for sbatch/srun) ---"
if command -v accounts &>/dev/null; then
    accounts
    echo "GPU jobs: DELTA_ACCOUNT=${DELTA_ACCOUNT:-bgpu-delta-gpu} (set in delta_env.sh)"
else
    echo "Run 'accounts' to find your billing account"
fi

echo ""
echo "Setup complete. Smoke test:"
echo "  srun --account=${DELTA_ACCOUNT:-bgpu-delta-gpu} --partition=gpuA100x4-interactive \\"
echo "    --gpus-per-node=1 --cpus-per-task=4 --mem=32g --time=00:30:00 \\"
echo "    bash experiments/GIFNO/delta_train.sh --limit 50"
