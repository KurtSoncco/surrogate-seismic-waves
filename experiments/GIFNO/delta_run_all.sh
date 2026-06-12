#!/bin/bash
# End-to-end Delta GIFNO setup from local WSL.
#
# Run in YOUR terminal (needs sudo for Box mount + Duo for Delta SSH):
#   bash experiments/GIFNO/delta_run_all.sh
#
# Steps: mount Box → rsync data → clone repo on Delta → install deps → W&B → smoke test

set -euo pipefail

DELTA_USER="${DELTA_USER:-ksonccosinchi}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
DELTA_ALLOC="${DELTA_ALLOC:-bgpu}"
export DELTA_ALLOC
LOCAL_DATA="/mnt/box_lab/Projects/Neural Operator/data"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIFNO_DIR="${PROJECT_ROOT}/experiments/GIFNO"
# Delta home is /u/$USER — there is no /scratch/$USER
REMOTE_REPO="~/surrogate-seismic-waves"

step() { echo ""; echo "========== $* =========="; }

# --- 1. Mount Box ---
step "1/6  Mount Box"
if [[ ! -d "${LOCAL_DATA}/h5" ]]; then
    echo "Mounting Box (may ask for sudo password)..."
    sudo mount -t drvfs "C:\\Users\\kurt-\\Box\\GIG Lab - UC Berkeley" /mnt/box_lab 2>/dev/null || true
fi
if [[ ! -d "${LOCAL_DATA}/h5" ]]; then
    echo "ERROR: Box data not at ${LOCAL_DATA}. Run: go-lab" >&2
    exit 1
fi
echo "H5 files local: $(ls "${LOCAL_DATA}/h5"/run_*.h5 2>/dev/null | wc -l)"

# --- 2. Rsync data ---
step "2/6  Rsync data to Delta (Duo MFA on first connection)"
bash "${GIFNO_DIR}/delta_rsync_from_local.sh"

# --- 3. Clone repo + copy secrets on Delta ---
step "3/6  Clone repo on Delta"
ssh "${DELTA_USER}@${DELTA_HOST}" bash -s <<REMOTE
set -euo pipefail
cd ~
if [[ -d surrogate-seismic-waves/.git ]]; then
    cd surrogate-seismic-waves && git pull
else
    git clone https://github.com/KurtSoncco/surrogate-seismic-waves.git
fi
REMOTE

step "4/6  Copy Delta scripts + W&B secrets"
rsync -avz "${GIFNO_DIR}/delta_train.sh" "${GIFNO_DIR}/delta_setup.sh" \
    "${DELTA_USER}@${DELTA_HOST}:${REMOTE_REPO}/experiments/GIFNO/"
if [[ -f "${GIFNO_DIR}/lambda_secrets.env" ]]; then
    scp "${GIFNO_DIR}/lambda_secrets.env" \
        "${DELTA_USER}@${DELTA_HOST}:${REMOTE_REPO}/experiments/GIFNO/"
fi

# --- 4. Setup env on Delta ---
step "5/6  Install deps + W&B on Delta"
ssh "${DELTA_USER}@${DELTA_HOST}" "cd ${REMOTE_REPO} && bash experiments/GIFNO/delta_setup.sh"

# --- 5. Smoke test on GPU ---
step "6/6  Smoke test (interactive GPU, --limit 50)"
echo "Fetching Delta account..."
DELTA_ACCOUNT="$(ssh "${DELTA_USER}@${DELTA_HOST}" "accounts 2>/dev/null | awk '/^[a-z]/ {print \$1; exit}'" || true)"
if [[ -z "${DELTA_ACCOUNT}" ]]; then
    echo "Could not auto-detect account. Run manually:"
    echo "  ssh ${DELTA_USER}@${DELTA_HOST}"
    echo "  accounts   # pick your allocation"
    echo "  cd ${REMOTE_REPO}"
    echo "  srun --account=<ACCOUNT> --partition=gpuA100x4-interactive \\"
    echo "    --gpus-per-node=1 --cpus-per-task=4 --mem=32g --time=00:30:00 \\"
    echo "    bash experiments/GIFNO/delta_train.sh --limit 50"
    exit 0
fi

echo "Using account: ${DELTA_ACCOUNT}"
ssh "${DELTA_USER}@${DELTA_HOST}" bash -s <<REMOTE
set -euo pipefail
cd ${REMOTE_REPO}
srun --account=${DELTA_ACCOUNT} --partition=gpuA100x4-interactive \
    --gpus-per-node=1 --cpus-per-task=4 --mem=32g --time=00:30:00 \
    bash experiments/GIFNO/delta_train.sh --limit 50
REMOTE

echo ""
echo "All done. Full training:"
echo "  ssh ${DELTA_USER}@${DELTA_HOST}"
echo "  cd ${REMOTE_REPO}/experiments/GIFNO"
echo "  # Edit #SBATCH --account= in delta_train.sh, then:"
echo "  sbatch delta_train.sh"
