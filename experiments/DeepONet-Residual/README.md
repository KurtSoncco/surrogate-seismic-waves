# DeepONet-Residual

Signed residual DeepONet: **single shared branch** encodes material fields
`(Vs, ζ, Z=ρ·Vs)` plus stochastic context `(ξ_re/im, r_H, aHV, CoV, ξ_damp)`,
trunk queries mesh-agnostic `(x/λ, f*, sin, cos)`, output signed
`R = TF_2D − TF_1D` for **R_col** vs **R_nom**.

Architecture choice follows Park et al. (Commun Eng 2026 / arXiv:2507.03660):
single-branch preferred for tightly coupled multiphysics; multi-branch kept as ablation.

Feature screening (MI/RF gate) lives in sibling [`../Residual/`](../Residual/).

## Training

- Loss: `SmoothL1Loss` only (`beta=1.0` in config)
- Optimizer: `AdamW` with `betas=(0.9, 0.999)`, `weight_decay=1e-5`

## Quick start

```bash
# Build signed cache (reuses Residual sample indices)
uv run python experiments/DeepONet-Residual/residual_signed.py --cache-tag n100_seed42

# Train default single-branch
uv run python experiments/DeepONet-Residual/train.py \
  --cache-tag n100_seed42 --target R_col --branch-mode single --trunk-set full

# Ablation sweep
uv run python experiments/DeepONet-Residual/run_ablation.py --cache-tag n100_seed42 --epochs 30
```

## Layout

| File | Role |
|------|------|
| `residual_signed.py` | Cache signed R + TF_1D baselines |
| `model.py` | SingleBranch / MultiBranch DeepONet |
| `data.py` | Field + stochastic + trunk dataset |
| `train.py` | Train / eval (R and TF reconstruction) |
| `run_ablation.py` | Branch / trunk / target sweep |

Results: `results/`; checkpoints: `checkpoints/`.
