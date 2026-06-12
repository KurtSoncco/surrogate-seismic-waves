#!/bin/bash
# One-time environment setup on NCSA Delta (run on login node).
#
#   ssh ksonccosinchi@login.delta.ncsa.illinois.edu
#   cd $SCRATCH && git clone https://github.com/KurtSoncco/surrogate-seismic-waves.git
#   cd surrogate-seismic-waves && bash experiments/GIFNO/delta_setup.sh
#
# After setup, copy W&B secrets from local:
#   scp experiments/GIFNO/lambda_secrets.env \
#     ksonccosinchi@login.delta.ncsa.illinois.edu:$SCRATCH/surrogate-seismic-waves/experiments/GIFNO/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"
DATA_ROOT="${GIFNO_DATA_ROOT:-${SCRATCH:-${HOME}}/gifno_data}"

echo "=== Delta GIFNO setup ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SCRATCH=${SCRATCH:-unset}"
echo "========================="

module purge
module load python/3.11
module load cuda/12.4

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${PROJECT_ROOT}"
uv venv .venv --python 3.11
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
echo "--- Delta account (for sbatch/srun) ---"
if command -v accounts &>/dev/null; then
    accounts
    echo "Set DELTA_ACCOUNT in delta_train.sh #SBATCH --account= line"
else
    echo "Run 'accounts' to find your allocation name"
fi

echo ""
echo "Setup complete. Smoke test:"
echo "  srun --account=<YOUR_ACCOUNT> --partition=gpuA100x4-interactive \\"
echo "    --gpus-per-node=1 --cpus-per-task=4 --mem=32g --time=00:30:00 \\"
echo "    bash experiments/GIFNO/delta_train.sh --limit 50"
