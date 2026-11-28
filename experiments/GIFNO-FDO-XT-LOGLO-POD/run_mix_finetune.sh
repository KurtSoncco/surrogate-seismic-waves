#!/usr/bin/env bash
# Pretrain on the IID + OOD + section mix, then fine-tune on Case-7-like sections
# (plan step 8: MIFNO transfer, Poseidon pretrain-then-fine-tune).
#
#   STAGE=pretrain  build DATA_OUT/mix_pretrain: IID share IID_SHARE (0.34) next
#                   to dipping, three-layer and every SECTION_DIRS entry, 10 %
#                   held out (sections by run); train one member per SEEDS.
#   STAGE=finetune  build DATA_OUT/mix_finetune: all section windows plus an
#                   IID/OOD replay share (REPLAY_SHARE, 0.1); warm-start every
#                   member from CK_OUT/pretrain_s<seed> at FT_LR for FT_EPOCHS.
#   STAGE=all       both.
#
#   IID_DIR=.../transfer_function DIP_DIR=.../ood_dipping TL_DIR=.../ood_three_layer \
#   CORNER_DIR=.../corner_is \
#   SECTION_DIRS="$SCRATCH/case7_sections/windows_round0 $SCRATCH/case7_sections/windows_round1" \
#   DATA_OUT=$WORK/outcrop_tf CK_OUT=$WORK/checkpoints SEEDS="0 1 2" STAGE=all \
#   bash run_mix_finetune.sh
#   TF_TARGET=outcrop ... bash run_mix_finetune.sh   # absolute TF head
#   bash run_dual_head.sh                            # residual then outcrop
#
# IID/DIP/TL/CORNER dirs are relabel_outcrop.py outputs, SECTION_DIRS are
# section_windows.py outputs. The members trained here are the deep ensemble
# that acquire_sections.py and score_section.py take as --model-dir.
#
# Pretrain fits the shared POD on this mix (GIFNO_REBUILD_POD=1) so the basis
# and LOGLO see every campaign, H, and Lx — not an IID-only linear residual.
# Train both heads: TF_TARGET=residual and TF_TARGET=outcrop (run_dual_head.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
: "${DATA_OUT:?set DATA_OUT}" "${CK_OUT:?set CK_OUT}"
STAGE="${STAGE:-all}"
SEEDS="${SEEDS:-0 1 2}"
TF_TARGET="${TF_TARGET:-residual}"
IID_SHARE="${IID_SHARE:-0.34}"
REPLAY_SHARE="${REPLAY_SHARE:-0.1}"
FT_LR="${FT_LR:-2e-4}"
FT_EPOCHS="${FT_EPOCHS:-60}"
read -r -a SECTIONS <<< "${SECTION_DIRS:-}"

sources=()
[[ -n "${IID_DIR:-}" ]] && sources+=(--source "iid=${IID_DIR}")
[[ -n "${DIP_DIR:-}" ]] && sources+=(--source "ood_dipping=${DIP_DIR}")
[[ -n "${TL_DIR:-}" ]] && sources+=(--source "ood_three_layer=${TL_DIR}")
[[ -n "${CORNER_DIR:-}" ]] && sources+=(--source "corner_is=${CORNER_DIR}")
section_sources=()
for i in "${!SECTIONS[@]}"; do section_sources+=(--source "sections_${i}=${SECTIONS[$i]}"); done

train_member() {  # data_dir model_dir seed [extra env...]
    local data="$1" model="$2" seed="$3"
    shift 3
    env "$@" OUTCROP_TF_DIR="$data" MODEL_DIR="$model" TF_TARGET="$TF_TARGET" GIFNO_SEED="$seed" \
        WANDB_RUN_NAME="$(basename "$model")" PYTHON="$PY" bash "$SCRIPT_DIR/run_abs_train.sh"
    if [[ -d "$data/holdout" && -f "$model/best_model.pt" ]]; then
        echo "=== holdout Pearson $model ==="
        "$PY" "$SCRIPT_DIR/score_holdout_pearson.py" \
            --model-dir "$model" --data "$data/holdout" --out "$model/holdout_pearson.json" \
            --arm "$(basename "$model")"
    fi
}

if [[ "$STAGE" == pretrain || "$STAGE" == all ]]; then
    "$PY" "$SCRIPT_DIR/build_outcrop_dataset.py" --target "$TF_TARGET" "${sources[@]}" "${section_sources[@]}" \
        --share "iid=${IID_SHARE}" --holdout-frac 0.1 --out "$DATA_OUT/mix_pretrain"
    for s in $SEEDS; do
        # Fit shared POD on this mix (IID + OOD + sections), not an old IID basis.
        train_member "$DATA_OUT/mix_pretrain" "$CK_OUT/pretrain_s$s" "$s" GIFNO_REBUILD_POD=1
    done
fi

if [[ "$STAGE" == finetune || "$STAGE" == all ]]; then
    (( ${#SECTIONS[@]} )) || { echo "finetune needs SECTION_DIRS" >&2; exit 1; }
    replay=()
    for spec in "${sources[@]}"; do
        [[ "$spec" == --source ]] && continue
        replay+=(--source "$spec" --share "${spec%%=*}=$(awk -v r="$REPLAY_SHARE" -v n="$(( ${#sources[@]} / 2 ))" 'BEGIN{print r/n}')")
    done
    "$PY" "$SCRIPT_DIR/build_outcrop_dataset.py" --target "$TF_TARGET" "${section_sources[@]}" "${replay[@]}" \
        --holdout-frac 0.1 --out "$DATA_OUT/mix_finetune"
    for s in $SEEDS; do
        pre="$CK_OUT/pretrain_s$s"
        ft="$CK_OUT/finetune_s$s"
        mkdir -p "$ft"
        cp "$pre/pod_modes.npy" "$pre/pod_mean.npy" "$ft/"
        train_member "$DATA_OUT/mix_finetune" "$ft" "$s" \
            GIFNO_INIT_CHECKPOINT="$pre/best_model.pt" GIFNO_LEARNING_RATE="$FT_LR" GIFNO_NUM_EPOCHS="$FT_EPOCHS"
    done
fi
