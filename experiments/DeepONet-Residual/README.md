# DeepONet-Residual

Signed residual DeepONet: **single shared branch** encodes material fields
`(Vs, ζ, Z=ρ·Vs)` plus stochastic context `(ξ_re/im, r_H, aHV, CoV, ξ_damp)`,
trunk queries mesh-agnostic `(x/λ, f*, sin, cos)` and (shipped) `log TF_1D`,
output signed `R = TF_2D − TF_1D` for geometry-aware **R_nom**.

**Shipped recipe** (RESULTS.md §9): geometry-aware Haskell nom + mixed-domain
train (IID + dipping + three_layer) + serial TF₁D-conditioned ResUNet.
Checkpoint: `checkpoints/arch_serial_P3_mix.pt`.

Architecture choice follows Park et al. (Commun Eng 2026 / arXiv:2507.03660):
single-branch preferred for tightly coupled multiphysics; multi-branch kept as ablation.

Feature screening (MI/RF gate) lives in sibling [`../Residual/`](../Residual/).

## Training

- Loss: `SmoothL1Loss` only (`beta=1.0` in config)
- Optimizer: `AdamW` with `betas=(0.9, 0.999)`, `weight_decay=1e-5`
- Train on `--n-freq-train` log-spaced queries (default **200**); **always eval at 1000 bins**
- Defaults: `--target R_nom --field-encoder resunet --serial-tf1d`

## Quick start

```bash
# Domain-mix study (splits, operator bake-off, P0–P4, arch). Caches reused if present.
GIFNO_DATA_ROOT=data/gifno_screen \
GIFNO_OOD_DIPPING=data/gifno_screen/ood_dipping \
GIFNO_OOD_THREE_LAYER=data/gifno_screen/ood_three_layer \
uv run python experiments/DeepONet-Residual/domain_study.py --skip-cache

# Score shipped residual on Box OOD (default checkpoint = serial P3 mix)
uv run python experiments/DeepONet-Residual/eval_ood.py

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

## Layout

| File | Role |
|------|------|
| `residual_signed.py` | Stratified indices + signed R + TF_1D baselines |
| `model.py` | SingleBranch / MultiBranch DeepONet |
| `data.py` | Field + stochastic + trunk dataset |
| `train.py` | Train / eval (R and TF reconstruction) |
| `domain_study.py` | Operator / P0–P4 mix / architecture bake-off |
| `run_scale.py` | IID `cache_tag × encoder × n_freq × seed` |
| `eval_ood.py` | Box `ood_*` Haskell nom/col ± R̂ (default: shipped ckpt) |
| `probe_ood.py` | Inventory OOD tree / attrs |
| `stage_screen_pack.sh` | Copy TF + stratified H5 + OOD |
| `run_ablation.py` | Branch / trunk / target sweep |

Results: `results/`; checkpoints: `checkpoints/`.
