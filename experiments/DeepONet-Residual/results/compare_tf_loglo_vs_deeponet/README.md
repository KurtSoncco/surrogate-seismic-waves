# TF head-to-head: LOGLO-POD vs ResUNet DeepONet (R_nom)

Same **1152** DeepONet `full7680_seed42` test samples, **full 1000 frequencies**.

| Model | mean rel L2 ↓ | mean Pearson_f ↑ | mean R²_TF ↑ | win rate (rel L2) |
|-------|--------------:|-----------------:|-------------:|------------------:|
| **LOGLO-POD full7680** | **0.242** | **0.945** | **0.880** | **95.5%** |
| DeepONet ResUNet (TF₁D+R̂) | 0.356 | 0.892 | 0.745 | 4.5% |
| TF₁D_nom only (baseline) | 0.494 | 0.768 | 0.514 | — |

**Verdict:** On in-distribution TF fidelity, **LOGLO-POD is clearly better**. DeepONet still beats the 1D Haskell baseline (ΔR² ≈ +0.23 vs TF1D) but loses to LOGLO on almost every sample.

Note: DeepONet is a residual operator (mesh-agnostic trunk); LOGLO is a direct spectral surrogate. Different inductive biases — this table is TF accuracy only.

**OOD comparison:** [`compare_tf_ood_loglo_vs_deeponet/README.md`](compare_tf_ood_loglo_vs_deeponet/README.md) (1920 dipping + three_layer cases).
