# Residual GINO evaluation: model, metrics, and SOTA comparison

Nested held-out scoring of the shipped residual GINO against OpenSees 2-D and classical 1-D site-response arms. LOGLO is out of scope.

**Checkpoint.** `experiments/DeepONet-Residual/checkpoints/M7680_gino_rebal_ft.pt`  
**Tests.** Nested IID \(n=150\), dipping \(n=144\), three-layer \(n=144\) ([`DATA.md`](../../../DATA.md)).  
**Val.** Same 70/15/15 splits: IID \(n=150\), dipping \(n=144\). The TF atlas and Sobol occupancy figures use **val+test** (IID 300, dipping 288). Paper rankings stay on frozen **test**.  
**Figures.** [`results/README.md`](../../README.md) lists the canonical PNGs.

---

## 1. Model

Residual GINO predicts the leftover

\[
R(x,f)=\mathrm{TF}_{2D}(x,f)-\mathrm{TF}_{1D}(x,f),
\]

then reconstructs \(\widehat{\mathrm{TF}}=\mathrm{TF}_{1D}+\hat R\). \(\mathrm{TF}_{1D}\) is geometry-aware Thomson–Haskell on the **nominal** 1-layer column (two-layer soil+bedrock; three-layer soil is *not* used for the 1-D backbone, so the three-layer test is misspecified 1-D by design).

| Piece | Shipped setting |
|-------|-----------------|
| Mix | M7680, `iid_frac=0.34`, three-layer val stop |
| Recipe | freeze-GNO fine-tune of `M700_gino.pt` |
| Encoder | recorder GNO (1-D depth conv + 3-layer kNN=2 message passing) |
| Residual mixer | vanilla FNO-on-\(R\) (4 FNOBlocks, modes 8×16) |
| Branch | single shared branch on \((V_s,\zeta,Z=\rho V_s)\) plus **`xi_cov`**: ξ (KL of the GRF PSD; mode ranking uses \(r_H,a_{HV}\)) and scalar CoV. Scalar \(r_H\)/\(a_{HV}\) are not explicit channels |
| Trunk | mesh-agnostic \((x/\lambda,f^*,\sin,\cos)\) and serial \(\log\mathrm{TF}_{1D}\) |
| Target | signed \(R_{\mathrm{nom}}\) |
| Train queries | 200 log-spaced frequencies; eval always at 1000 bins (0.1–10 Hz) |
| Loss | Smooth L1 (\(\beta=1\)) |

Training JSON: `results/arch_train/M7680_gino_rebal_ft.json`. On the same nested indices the checkpoint reports Pearson of \(|\mathrm{TF}|\) vs OpenSees of **0.926** (IID), **0.903** (dipping), **0.869** (three-layer), and leftover \(R^2\) of 0.61 / 0.70 / 0.47.

---

## 2. Metrics

All method rankings and covariate plots use two **primary** scores on the central recorder, versus OpenSees 2-D:

| Metric | Definition | Better |
|--------|------------|--------|
| **Pearson** | Pearson correlation of \(\lvert\mathrm{TF}\rvert\) vs frequency | higher |
| **Anderson misfit** | weighted \(L_1\) on \(\ln\lvert\mathrm{TF}\rvert\) | lower |

**All-band Anderson** is Gaussian-weighted at the **window-extracted** 2-D fundamental \(f_0\): tallest `scipy.signal.find_peaks` \(\lvert\mathrm{TF}\rvert\) in a log-symmetric ±20% window around the travel-time guess \(1/(4T)\). That \(1/(4T)\) value is only the search-window center (so harmonics are not picked as \(f_0\)). **Band Anderson** is uniform-weight \(L_1\) inside the band.

Frequency bands: low 0.1–0.5 Hz, mid 0.5–2 Hz, high 2–10 Hz, all 0.1–10 Hz.

**Secondary** (tables / leftover figures only): relative \(L_2\), leftover slope \(b\) from \(\hat R=a+bR\) (trough-windowed OLS, cluster-bootstrap CI), \(\Delta\ln A_{\mathrm{peak}}\), trough-safe log bias. \(b=1\) is a perfect leftover; \(b<1\) is under-correction.

---

## 3. Classical arms (one Pretell pair)

There are **two Pretell quantities**, not three:

| Display name | What it is |
|--------------|------------|
| **Pretell median** | Geometric mean of Thomson–Haskell \(\lvert\mathrm{TF}\rvert\) on 200 columns across the 500 m variability strip. For a lognormal envelope this geomean is the median. |
| **Pretell percentile** | Same median \(\times\exp(\sigma_{\ln})\), i.e. the ~84th-percentile \(\lvert\mathrm{TF}\rvert(f)\) envelope. |

An older cache series, per-recorder column Haskell (`tf_haskell_column`), used to be plotted as “Pretell's approach”. That is **not** the 200-column Pretell arm; it is excluded from ranking whenever Pretell median is present.

Other arms:

| Arm | Predictor |
|-----|-----------|
| OpenSees 2-D | Ground truth |
| GINO | \(\mathrm{TF}_{1D}+\hat R\) |
| 1D Base Case | Nominal Haskell (same backbone GINO residualizes) |
| Toro Vs | seiskit simplified Vs-rand, 40 seeds, geomean |
| Passeri tts | seiskit TTS-rand, 40 seeds, geomean |
| Dmult | seiskit Hallal Approach 5 (`hallal_dmin`): base-case \(V_s\), elemental Q–Vs damping, 10 multipliers `linspace(3, 6, 10)`, geomean |

Dmult p84 exists as an envelope (same construction as Pretell percentile) but is **not** a primary ranking arm.

On dipping, the 1-layer nom is a valid 1-D sketch of a dipping interface. On **three-layer**, that 1-layer nom is misspecified by design: Toro / Passeri / Dmult still randomize a single soil layer, so they should collapse. Pretell still sees the 2-D \(V_s\) field through 200 columns.

---

## 4. Leftover learning (GINO vs OpenSees)

OLS \(\hat R\) on \(R\), central recorder, trough-windowed, cluster-bootstrap 95% CI:

| Domain | \(b\) | 95% CI | Reading |
|--------|------:|--------|---------|
| IID | 0.557 | [0.403, 0.742] | under-correction |
| Dipping | 0.712 | [0.685, 0.737] | under-correction |
| Three-layer | 0.555 | [0.510, 0.616] | under-correction |

GINO recovers more than half of the 2-D leftover on every nested slice, most on dipping, and does not overshoot (\(b<1\)). Mean Pearson / Anderson of the **reconstructed** \(\lvert\mathrm{TF}\rvert\):

| Domain | \(n\) | Pearson mean (median) | Anderson mean (median) |
|--------|------:|----------------------:|-----------------------:|
| IID | 150 | 0.926 (0.943) | 0.111 (0.098) |
| Dipping | 144 | 0.912 (0.928) | 0.111 (0.101) |
| Three-layer | 144 | 0.878 (0.914) | 0.183 (0.167) |

Window-extracted \(f_0\) spans ~0.28–4.3 Hz (IID), ~0.60–2.3 Hz (dipping), ~1.8–7.2 Hz (three-layer).

Figures: `leftover_calibration.png`, `leftover_vs_freq.png`, presentation column (c) on `compare_*_page*.png`.

---

## 5. SOTA ranking

Primary scores. Pearson higher is better; Anderson lower is better. IID uses means over \(n=150\); OOD compact ranking uses **medians** with bootstrap CIs (`ood_method_ranking.csv`).

### 5.1 Nested IID (\(n=150\))

| Method | Pearson (mean) | Anderson (mean) | Pearson low / mid / high | Anderson low / mid / high |
|--------|---------------:|----------------:|--------------------------|---------------------------|
| **GINO** | **0.926** | 0.111 | 0.751 / **0.939** / **0.794** | 0.066 / 0.126 / 0.323 |
| Pretell median | 0.906 | **0.093** | **0.997** / 0.925 / 0.763 | **0.010** / **0.118** / **0.319** |
| Pretell percentile | 0.897 | 0.156 | 0.995 / 0.910 / 0.754 | 0.033 / 0.210 / 0.359 |
| Toro Vs | 0.784 | 0.175 | 0.982 / 0.768 / 0.517 | 0.030 / 0.247 / 0.400 |
| Passeri tts | 0.754 | 0.178 | 0.972 / 0.758 / 0.550 | 0.040 / 0.241 / 0.407 |
| 1D Base Case | 0.746 | 0.179 | 0.970 / 0.754 / 0.534 | 0.041 / 0.242 / 0.414 |
| Dmult | 0.673 | 0.277 | 0.975 / 0.702 / 0.436 | 0.070 / 0.368 / 1.209 |

**IID reading.** GINO is the best Pearson match to OpenSees (0.926 vs Pretell median 0.906). Pretell median is slightly better on Anderson (0.093 vs 0.111), especially below 0.5 Hz where a 200-column geomean is almost a perfect low-frequency shape. GINO pulls ahead in the mid and high bands (Pearson 0.939 / 0.794 vs 0.925 / 0.763). Toro and Passeri sit with the 1-D base case. Dmult (damping-only sweep, no Vs randomization) is last: it under-predicts peak amplitude (\(\mathrm{median}\,\Delta\ln A\approx-1.30\)).

Pretell percentile is an **upper envelope**, not a competing median predictor. It is close in Pearson (0.897) but worse in Anderson (0.156) because the \(+\sigma_{\ln}\) scale bias is large near resonance.

### 5.2 Compact OOD (median Pearson / median Anderson)

| Method | Dipping Pearson | Dipping Anderson | Three-layer Pearson | Three-layer Anderson |
|--------|----------------:|-----------------:|--------------------:|---------------------:|
| **GINO** | **0.928** [0.916, 0.940] | **0.101** [0.094, 0.107] | **0.914** [0.895, 0.931] | **0.167** [0.145, 0.190] |
| Pretell median | 0.758 [0.736, 0.794] | 0.119 [0.106, 0.129] | 0.785 [0.752, 0.830] | 0.235 [0.206, 0.250] |
| Pretell percentile | 0.754 [0.737, 0.800] | 0.171 [0.162, 0.177] | 0.789 [0.755, 0.823] | 0.285 [0.257, 0.342] |
| Toro Vs | 0.705 [0.688, 0.740] | 0.133 [0.126, 0.139] | 0.026 [−0.006, 0.047] | 0.879 [0.813, 0.919] |
| 1D Base Case | 0.690 [0.670, 0.725] | 0.138 [0.129, 0.148] | 0.673 [0.614, 0.730] | 0.279 [0.243, 0.311] |
| Passeri tts | 0.685 [0.663, 0.711] | 0.140 [0.129, 0.149] | −0.010 [−0.029, −0.002] | 0.984 [0.923, 1.022] |
| Dmult | 0.505 [0.466, 0.538] | 0.248 [0.240, 0.258] | −0.017 [−0.079, 0.011] | 0.887 [0.812, 0.995] |

**OOD reading.** Geometry is where GINO earns its keep. On dipping, median Pearson 0.928 vs Pretell 0.758 and 1-D 0.690. On three-layer, GINO 0.914 vs Pretell 0.785; Toro / Passeri / Dmult fall to ~0 Pearson because they still randomize a **single** soil layer on a two-layer soil column. Pretell does not collapse there because it samples the actual 2-D field. The 1-D base case (0.67 Pearson) is the misspecified nom GINO residualizes, so GINO can still add \(\hat R\) on top of a wrong 1-D skeleton and recover the 2-D shape.

Figures: `method_ranking_iid.png`, `method_ranking_iid_bands.png`, `method_ranking_ood_compact.png` (nested **test**). Val+test Pearson boxes, IID and dipping side by side: [`method_ranking_pearson_heldout.png`](method_ranking_pearson_heldout.png) (\(n=300\) / \(288\)). Importance-sampled corner OpenSees (32×5, shipped GINO never trained on these): [`method_ranking_pearson_corner.png`](method_ranking_pearson_corner.png) (24 train-eligible locations vs 8 held-out).

### 5.3 Realization ceiling (GINO only; appendix / discussion)

OpenSees–OpenSees Pearson of \(|\mathrm{TF}|\) at fixed 6D θ is the ceiling for a θ-only predictor (median 0.83 IID / 0.85 dipping on replicated nested-test cells). GINO exceeds that self-agreement in **28/28** IID and **29/30** dipping cells (median gap +0.10). Report that fraction with Pearson/Anderson; do not treat Pearson\(<0.9\) as a failure rate. Detail, sample 50, and the combinatorially empty joint corner: [`TAIL_A_VS_B.md`](TAIL_A_VS_B.md).

---

## 6. Covariate bias (GINO only)

Quartile bins and Spearman \(\rho\) of Pearson and Anderson versus H5 parameters, using window-extracted \(f_0\) (not \(V_{s1}/(4H)\)). Shared axes: \(V_{s1}\), \(H\), CoV, \(V_{s2}\), \(r_H\), \(a_{HV}\), extracted \(f_0\), impedance \(V_{s2}/V_{s1}\), soil-column mean \(\zeta\), empirical field CoV of cropped \(V_s\). Dipping extras: dip angle, dip span, bedrock thickness. Three-layer extras: \(H_1\), \(H_2\), \(V_{s,\mathrm{mid}}\), \(\ln(V_{s,\mathrm{mid}}/V_{s1})\).

Spearman of **all-band** Pearson / Anderson vs selected axes:

| Covariate | IID \(\rho\) Pearson / Anderson | Dipping | Three-layer |
|-----------|--------------------------------:|---------|-------------|
| CoV | −0.49 / +0.47 | −0.64 / +0.60 | −0.35 / +0.47 |
| Field CoV | −0.49 / +0.47 | −0.40 / +0.59 | −0.13 / +0.03 |
| \(r_H\) | −0.31 / +0.27 | −0.45 / +0.44 | (little spread) |
| \(H\) | +0.11 / −0.16 | +0.11 / −0.34 | −0.02 / −0.22 |
| Extracted \(f_0\) | −0.09 / +0.10 | +0.01 / +0.06 | +0.07 / +0.18 |
| \(V_{s1}\) | ~0 / −0.12 | +0.10 / −0.22 | +0.10 / +0.02 |

**Bias reading.** The leftover GINO still struggles when the random field is rough: higher prescribed CoV and higher empirical field CoV both lower Pearson and raise Anderson, on IID and especially dipping. Horizontal correlation \(r_H\) is the next geometric axis (longer \(r_H\) is slightly harder). Site frequency and \(V_{s1}\) are weak all-band predictors once \(f_0\) is the extracted peak rather than \(1/(4T)\). Mean \(\zeta\) from `Damping_zeta` has little spread (std ≈ 0.006) and is not a useful axis.

Figures: `bias_vs_covariates.png`, `bias_vs_covariates_anderson.png`, `bias_vs_covariates_bands.png`, `bias_quartile_forest.png`. Tables: `gino_bias/per_sample.csv`, `gino_bias/covariate_spearman.csv`.

---

## 7. Summary

1. **Pretell is one method with two plotted statistics:** median (200-column Haskell geomean) and percentile (\(\mathrm{median}\times e^{\sigma_{\ln}}\)). The old “Pretell's approach” label was per-recorder column Haskell and is no longer ranked.

2. **On nested IID, GINO and Pretell median are close.** GINO wins Pearson of \(\lvert\mathrm{TF}\rvert\) (0.926 vs 0.906). Pretell median wins Anderson (0.093 vs 0.111), mostly from an almost exact low-frequency geomean. Toro, Passeri, and the 1-D nom cluster together; Dmult is not competitive as a \(\lvert\mathrm{TF}\rvert\) shape predictor.

3. **On dipping and three-layer, GINO is clearly ahead of every 1-D arm**, including Pretell median (median Pearson 0.93 / 0.91 vs 0.76 / 0.79). Randomized 1-layer Hallal arms are unusable on three-layer, as expected.

4. **The leftover is real but under-corrected** (\(b\approx 0.56\) IID and three-layer, \(0.71\) dipping). GINO does not invent extra 2-D motion; it returns a fraction of \(R=\mathrm{TF}_{2D}-\mathrm{TF}_{1D}\).

5. **Remaining GINO error tracks field roughness** (CoV, field CoV, \(r_H\)), not the quarter-wave \(f_0\) that used to be plotted as a covariate. That cloud is two mechanisms: a combinatorially empty high-CoV × high-\(r_H\) Sobol corner (\(0.2^2\times 256\approx 10\) predicted vs 13 IDs; 5-axis top-20% cell empty), and seed-to-seed OpenSees disagreement at fixed θ. GINO beats that θ-only ceiling in 28/28 IID and 29/30 dipping replicated cells ([`TAIL_A_VS_B.md`](TAIL_A_VS_B.md)).

6. **Pearson of \(\lvert\mathrm{TF}\rvert(f)\) is a shape score.** It is affine-invariant, so Toro / Passeri / Dmult can look aligned with OpenSees while peak amplitude is off (IID test: Toro mean Pearson 0.784 with median \(\Delta\ln A\approx-0.96\); Dmult 0.673 / \(\approx-1.30\)). Low-band Pearson is ~0.97 for every 1-D arm. Use Anderson and \(\Delta\ln A\) with Pearson; do not read a high all-band \(r\) as interchangeable methods. The 1-D Haskell nom is scored and plotted on the val+test atlas alongside GINO / Pretell / Toro / Passeri / Dmult.

7. **Val+test occupy the same Sobol box as train.** Nested IID train is 700 files / 242 unique 6-D IDs; val+test 300 / 187 unique, of which 177 overlap train (10 held-out-only IDs). Dipping train 672 / 33 unique 7-D points (6-D plus dip angle); val+test 288 / 32 unique, all inside train. Pairplots: [`pairplot_iid_6d.png`](../sobol_probe/pairplot_iid_6d.png), [`pairplot_dipping_7d.png`](../sobol_probe/pairplot_dipping_7d.png). The advisor gap is the empty joint high-CoV × high-\(r_H\) corner, not a val/test cloud outside the training hull. TF atlas (4×4, window-extracted \(f_0\)): [`tf_atlas/`](../tf_atlas/). Each panel is one RF seed (in the title) at the central recorder — not a spatial 16–84% band and not a mix of seeds.

Reproduce:

```bash
# caches: results/presentation/{iid,dipping,three_layer}_pack.npz
uv run python experiments/DeepONet-Residual/response_variability/diagnostics/analyze_gino_bias.py
uv run python experiments/DeepONet-Residual/response_variability/evals/eval_classical.py --domains dipping three_layer
uv run python experiments/DeepONet-Residual/response_variability/evals/eval_seiskit.py
uv run python experiments/DeepONet-Residual/response_variability/plots/plot_eval_bias.py
uv run python experiments/DeepONet-Residual/response_variability/sobol_cover.py
uv run python experiments/DeepONet-Residual/response_variability/plots/tf_atlas.py
uv run python experiments/DeepONet-Residual/response_variability/evals/eval_corner_is.py
```
