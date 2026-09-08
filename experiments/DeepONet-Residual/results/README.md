# Results layout (shipped models only)

Everything here supports **residual GINO rebal FT** (`M7680_gino_rebal_ft.pt`)
or its comparison to **LOGLO-POD full**. Ablation waves, ResUNet TF compares,
cov-only packs, and domain-study JSON are gone.

| Path | What it answers | How to regenerate |
|------|-----------------|-------------------|
| `presentation/` | Nature 2×3 case pages, Pearson histograms, Vs mosaic | `response_variability/plot_presentation.py` (`--skip-predict` uses `*_pack.npz`) |
| `presentation/{iid,dipping,three_layer}_pack.npz` | Cached OpenSees / GINO / Haskell / Pretell TFs for those pages (local only, gitignored) | `plot_presentation.py` (GPU) |
| `compare_gino_loglo/` | Nested-test rel L2: LOGLO vs GINO vs Haskell | See that folder’s README |
| `arch_train/M7680_gino_rebal_ft.json` | Shipped checkpoint metrics + ship gates | `arch_train.py` / `score_ship_gates.py` |
| `response_variability/seiskit/` | GINO vs Hallal / Toro / Passeri / Pretell on nested IID | `eval_iid.py` then `eval_seiskit.py` |
| `response_variability/sobol_probe/` | Covering, RV-64 proxy, train vs held-out frequency | `eval_sobol_probe.py` |
| `response_variability/gino_bias/` | Residual-bias instruments on the shipped ckpt | `analyze_gino_bias.py` |
| `response_variability/predictions.npz` | Held-out GINO TFs for the seiskit overlay | `eval_iid.py` |

**Canonical figures to cite**

- `presentation/pearson_histograms.png`
- `presentation/compare_{iid,dipping,three_layer}_page2.png` (median quantile page)
- `presentation/vs_mosaic.png`
- `response_variability/seiskit/tf_pearson.png`
- `response_variability/sobol_probe/{seiskit_arms_iid,cloud_rh_ahv,freq_train_vs_heldout}.png`
