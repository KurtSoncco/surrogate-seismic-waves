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
# Stratified indices (no Residual RF screen required)
uv run python experiments/DeepONet-Residual/select_indices.py --cache-tag n2000_seed42

# Signed R_nom cache (needs GIFNO_DATA_ROOT with h5/ + transfer_function/)
uv run python experiments/DeepONet-Residual/residual_signed.py --cache-tag n2000_seed42

# E1: ResUNet R_nom at n=2000, 50-freq train, 1000-freq eval
uv run python experiments/DeepONet-Residual/run_scale.py \
  --cache-tag n2000_seed42 --field-encoder resunet --target R_nom \
  --n-freq-train 50 --n-freq-eval 1000 --patience 60

# E0: Haskell nom vs col on Box ood_dipping / ood_three_layer
uv run python experiments/DeepONet-Residual/eval_ood.py

# Stage a 2k–3k pack on the laptop for a Box share link
bash experiments/DeepONet-Residual/stage_screen_pack.sh --n 3000
```

`eval_ood.py` defaults to `$GIFNO_DATA_ROOT/ood_dipping` and `ood_three_layer`
(not `~/seiskit/neural-operator/experiments`).

## Layout

| File | Role |
|------|------|
| `residual_signed.py` | Cache signed R + TF_1D baselines |
| `select_indices.py` | Nested stratified n=1000 ⊂ 2000 ⊂ 3000 |
| `model.py` | SingleBranch / MultiBranch DeepONet |
| `data.py` | Field + stochastic + trunk dataset |
| `train.py` | Train / eval (R and TF reconstruction) |
| `run_scale.py` | n × encoder × freq CLI |
| `eval_ood.py` | Haskell floor on Box OOD corpora |
| `stage_screen_pack.sh` | Copy TF + subset H5 + ood_* for sharing |
| `run_ablation.py` | Branch / trunk / target sweep |

Results: `results/`; checkpoints: `checkpoints/`.
