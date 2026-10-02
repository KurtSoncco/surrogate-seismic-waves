# Results layout (shipped models only)

Everything here supports **residual GINO rebal FT** (`M7680_gino_rebal_ft.pt`)
or its comparison to **LOGLO-POD full**. Ablation waves, ResUNet TF compares,
cov-only packs, and domain-study JSON are gone.

| Path | What it answers | How to regenerate |
|------|-----------------|-------------------|
| `presentation/` | Nature 2×3 case pages (leftover \(R\) vs \(\hat R\)), Pearson histograms, Vs mosaic | `response_variability/plots/plot_presentation.py` (`--skip-predict` uses `*_pack.npz`) |
| `presentation/{iid,dipping,three_layer}_pack.npz` | Cached OpenSees / GINO / Haskell / Pretell TFs for those pages (local only, gitignored) | `plot_presentation.py` (GPU) |
| `compare_gino_loglo/` | Nested-test rel L2: LOGLO vs GINO vs Haskell | See that folder’s README |
| `arch_train/M7680_gino_rebal_ft.json` | Shipped checkpoint metrics + ship gates | `arch_train.py` / `scoring/score_ship_gates.py` |
| `SPATIAL_QUERY.md` | What is a spatial query vs a Graph NO after kernel Savio (ship stays) | — |
| `arch_train/M7680_kernel_spatial_*.json` | Kernel GNO nested 21-station tests (three-layer KILL) | `hpc/savio_spatial_query.sh` |
| `arch_train/spatial_query_*.json` | Kernel station hold-out vs interpolate-p (after Savio spatial arms) | `scoring/eval_spatial_query.py` |
| `response_variability/seiskit/` | GINO vs Toro / Passeri / Pretell / **Dmult** on nested IID | `eval_iid.py` then `eval_seiskit.py` |
| `response_variability/eval_bias/` | Leftover calibration, Pearson/Anderson vs H5 covariates (all + bands), IID ranking, compact OOD ranking | `eval_classical.py` (OOD Toro/Passeri/Dmult) then `plot_eval_bias.py`; covariates via `analyze_gino_bias.py` |
| `response_variability/eval_bias/SUMMARY.md` | Model, metrics, Pretell median/percentile, nested SOTA tables | Written from the CSVs in this folder |
| `response_variability/eval_bias/TAIL_A_VS_B.md` | Pearson\(<0.9\) tail split: joint-corner occupancy vs seed-to-seed OpenSees ceiling; sample 50; dipping array-mean | `response_variability/diagnostics/tail_a_vs_b.py` |
| `response_variability/sobol_probe/` | Covering, RV-64 proxy, train vs held-out frequency, train vs val+test pairplots | `eval_sobol_probe.py`, `sobol_cover.py` |
| `response_variability/plots/tf_atlas/` | Val+test |TF| 4×4 pages sorted by window-extracted \(f_0\) (IID 300, dipping 288); 1-D nom scored with GINO / Pretell / Toro / Passeri / Dmult | `tf_atlas.py` |
| `response_variability/gino_bias/` | Residual-bias instruments on the shipped ckpt (Pearson/Anderson vs covariates) | `analyze_gino_bias.py` |
| `response_variability/predictions.npz` | Held-out GINO TFs for the seiskit overlay | `eval_iid.py` |

**Canonical figures to cite**

- `presentation/pearson_histograms.png`
- `presentation/compare_{iid,dipping,three_layer}_page2.png` (median quantile page; column c is leftover \(R\) vs \(\hat R\))
- `presentation/vs_mosaic.png`
- `response_variability/eval_bias/leftover_calibration.png`
- `response_variability/eval_bias/leftover_vs_freq.png`
- `response_variability/eval_bias/bias_vs_covariates.png` (Pearson) and `bias_vs_covariates_anderson.png`
- `response_variability/eval_bias/bias_vs_covariates_bands.png`
- `response_variability/eval_bias/bias_quartile_forest.png`
- `response_variability/eval_bias/method_ranking_iid.png` (and `method_ranking_iid_bands.png`)
- `response_variability/eval_bias/method_ranking_ood_compact.png`
- `response_variability/eval_bias/method_ranking_pearson_heldout.png` (Pearson boxes: IID val+test vs dipping val+test)
- `response_variability/eval_bias/method_ranking_pearson_corner.png` (Pearson boxes: IS corner train-eligible vs held-out)
- `response_variability/eval_bias/sample50_iid.png`
- `response_variability/eval_bias/recorder_sensitivity_dipping.png`
- `response_variability/seiskit/tf_pearson.png`
- `response_variability/sobol_probe/{seiskit_arms_iid,cloud_rh_ahv,freq_train_vs_heldout,pairplot_iid_6d,pairplot_dipping_7d}.png`
- `response_variability/plots/tf_atlas/{iid,dipping}/tf_panels_f0_quantiles.png` (full val+test pages in the same folders)
