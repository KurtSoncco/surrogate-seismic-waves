# DeepONet-Residual

Signed residual DeepONet: **single shared branch** encodes material fields
`(Vs, ζ, Z=ρ·Vs)` plus stochastic context `(ξ_re/im, r_H, aHV, CoV, ξ_damp)`,
trunk queries mesh-agnostic `(x/λ, f*, sin, cos)` and (shipped) `log TF_1D`,
output signed `R = TF_2D − TF_1D` for geometry-aware **R_nom**.

**Shipped recipe** (RESULTS.md §10): geometry-aware Haskell nom + mixed-domain
train (IID + dipping + three_layer) + serial TF₁D + recorder GNO + FNO-on-\(R\).
Checkpoint: `checkpoints/M700_gino.pt`. §9 control: `checkpoints/arch_serial_P3_mix.pt`.

Architecture choice follows Park et al. (Commun Eng 2026 / arXiv:2507.03660):
single-branch preferred for tightly coupled multiphysics; multi-branch kept as ablation.

Feature screening (MI/RF gate) lives in sibling [`../Residual/`](../Residual/).

## Training

- Loss: `SmoothL1Loss` only (`beta=1.0` in config); optional aux TF rel L2 in `arch_train.py`
- Optimizer: `AdamW` with `betas=(0.9, 0.999)`, `weight_decay=1e-5`
- Train on `--n-freq-train` log-spaced queries (default **200**); **always eval at 1000 bins**
- Defaults: `--target R_nom --field-encoder resunet --serial-tf1d`
- Progress: wandb + tqdm (offline if no `WANDB_API_KEY`; online on Lambda via `lambda_secrets.env`). §10–12 archive is `deeponet-residual`; n-scale Lambda runs go to `deeponet-nscale` (`lambda_train.sh` default).

## Quick start

```bash
# Domain-mix study (splits, operator bake-off, P0–P4, arch). Caches reused if present.
GIFNO_DATA_ROOT=data/gifno_screen \
GIFNO_OOD_DIPPING=data/gifno_screen/ood_dipping \
GIFNO_OOD_THREE_LAYER=data/gifno_screen/ood_three_layer \
uv run python experiments/DeepONet-Residual/domain_study.py --skip-cache

# Score shipped residual on Box OOD (default checkpoint = GINO)
uv run python experiments/DeepONet-Residual/eval_ood.py --split test

# n-ladder / FNO / GNO bake-off
uv run python experiments/DeepONet-Residual/arch_train.py --mix M700 --encoder gno --fno

# Haskell floor only
uv run python experiments/DeepONet-Residual/eval_ood.py --haskell-only
```

IID-only scale ladder (not the shipped mix):

```bash
uv run python experiments/DeepONet-Residual/residual_signed.py --cache-tag n2000_seed42
uv run python experiments/DeepONet-Residual/run_scale.py \
  --cache-tag n2000_seed42 --encoder resunet --n-freq-train 200
```

Stage a laptop/cloud pack (TF cache + stratified H5 + OOD):

```bash
experiments/DeepONet-Residual/stage_screen_pack.sh /path/to/gifno_screen
export GIFNO_DATA_ROOT=/path/to/gifno_screen
```

## Lambda (GINO-wide + wandb online)

Start a `gpu_1x_a10` (or A100). From the laptop:

```bash
HOST=ubuntu@<LAMBDA_IP>
rsync -az --exclude .venv --exclude wandb --exclude '*.pt' \
  ~/surrogate-seismic-waves/ "$HOST:~/surrogate-seismic-waves/"
rsync -az ~/surrogate-seismic-waves/data/gifno_screen/ \
  "$HOST:~/surrogate-seismic-waves/data/gifno_screen/"
rsync -az ~/surrogate-seismic-waves/experiments/DeepONet-Residual/cache/ \
  "$HOST:~/surrogate-seismic-waves/experiments/DeepONet-Residual/cache/"
scp experiments/GIFNO/lambda_secrets.env \
  "$HOST:~/surrogate-seismic-waves/experiments/GIFNO/"
```

On the instance:

```bash
tmux new-session -d -s gino \
  "cd ~/surrogate-seismic-waves && bash experiments/DeepONet-Residual/lambda_train.sh \
     --mix M2100 --encoder gno --fno --batch-size 32 \
     --fno-width 64 --fno-modes 8,32 --fno-layers 4 \
     --run-name M2100_gino_wide_lambda 2>&1 | tee train_gino.log"
```

Wandb: Lambda `lambda_train.sh` logs to project **`deeponet-nscale`** (tags `mix` / `encoder` / `fno_kind` / `host`). Leave `deeponet-residual` as the §10–12 archive. Override with `WANDB_PROJECT=...` if needed.

Sync laptop §10 offline wandb runs (needs `WANDB_API_KEY`):

```bash
wandb sync experiments/DeepONet-Residual/wandb/offline-run-*
```

## Layout

| File | Role |
|------|------|
| `residual_signed.py` | Stratified indices + signed R + TF_1D baselines |
| `model.py` | SingleBranch / MultiBranch / GNO DeepONet + FNO-on-R wrapper |
| `data.py` | Field + stochastic + trunk dataset |
| `train.py` | Train / eval (wandb + tqdm) |
| `arch_train.py` | n-ladder / recipe / FNO / GNO bake-off |
| `mix_ladder.py` | Nested-safe M700/M1400/M2100/M7680 mix indices |
| `domain_study.py` | Operator / P0–P4 mix / architecture bake-off |
| `lambda_train.sh` | Lambda Labs wrapper (wandb online, GINO-wide) |
| `run_scale.py` | IID `cache_tag × encoder × n_freq × seed` |
| `eval_ood.py` | Box `ood_*` Haskell nom/col ± R̂ (default: shipped ckpt) |
| `probe_ood.py` | Inventory OOD tree / attrs |
| `stage_screen_pack.sh` | Copy TF + stratified H5 + OOD |
| `run_ablation.py` | Branch / trunk / target sweep |

Results: `results/`; checkpoints: `checkpoints/`.
