# Residual / OrbitAll Feature Screen — Results

**Experiment:** `experiments/Residual/`
**Date:** 2026-08-03
**Primary run:** stratified subsample **n = 1000** (seed 42) of 7680 GIFNO OpenSees cases
**Pilot run:** n = 100 (same protocol; qualitative conclusions unchanged)

This experiment is a **hard gate**: if engineered geometric / physics features cannot explain variance in a simple Random Forest, they will not magically work in a downstream Graph Neural Operator. Raw spatial pixel coordinates are banned as inputs.

---

## 1. Data


| Artifact        | Location / shape                                                                  |
| --------------- | --------------------------------------------------------------------------------- |
| Source root     | `/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data`                    |
| OpenSees TF₂D   | `transfer_function/tf_per_sample.npy` — `(7680, 21, 1000)`                        |
| Frequencies     | `freq.npy` — 1000 log-spaced points in **[0.1, 10] Hz**                           |
| Recorders       | 21 lateral stations on the cropped 500 m variability strip (`recorder_x_idx.npy`) |
| Material fields | `h5/run_*.h5` — `Vs_realization_2D`, `Damping_zeta`, `params`                     |


Each sample’s random field was generated with an FFT spectral GRF (`seiskit.gaussian_field.generate_gaussian_field_fft`) controlled by `(rf_seed, rH, aHV, CoV, …)`. The wavy soil–rock interface amplitude in generation is **0**, so true basin dip is essentially flat for this corpus.

---



## 2. Targets (definitions)

Both targets are **magnitude residuals** between the 2D OpenSees transfer function and a 1D Thomson–Haskell baseline. They are evaluated at every recorder x_r and frequency f.

### $R_{col}$ — local-column residual (pure lateral coupling)

#
$\bigl|R_{\mathrm{col}}(x_r,f)\bigr|$

$\bigl|TF_{2D}(x_r,f) - TF_{1D,\mathrm{col}}(x_r,f)\bigr|$


- $TF_{2D}$: OpenSees surface/base TF magnitude (cached).
- $TF_{1D,\mathrm{col}}$: Haskell $|AF_{\mathrm{within}}|$ on the **realized** Vs/ζ column at that recorder (full depth layering, soil_nz from H5).

This isolates **lateral / 2D coupling** beyond what a perfect local 1D column already explains.

### R_{\mathrm{nom}} — nominal (no-variability) residual

#
$\bigl|R_{\mathrm{nom}}(x_r,f)\bigr|$

$\bigl|TF_{2D}(x_r,f) - TF_{1D,\mathrm{nom}}(f)\bigr|$


- TF_{1D,\mathrm{nom}}: single-layer Haskell from attrs `(Vs1, H, Vs2)` with fixed damping \xi = 0.05, **no** spatial RF.
- Broadcast across all recorders (same baseline curve at every x_r).

This residual mixes **(a)** RF / interlayer effects that a nominal 1D profile misses and **(b)** true 2D lateral coupling. Empirically |R_{\mathrm{nom}}| is larger than |R_{\mathrm{col}}| (see §6 plots).

Cached arrays (n=1000): `cache/n1000_seed42/r_col.npy`, `r_nom.npy` with shape `(1000, 21, 1000)`.

---



## 3. Feature dictionary (definitions)

No raw pixel (x,z) channels. Features are built per table row `(sample, recorder, freq)`.

### Geometrical


| Name            | Definition                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dip_slope`     | \nabla_x Z — lateral derivative of bedrock interface depth Z(x) from a Vs threshold (same idea as TH-FNO `interface_dip`).                             |
| `imp_grad`      | Lateral gradient of depth-mean soil impedance \partial_x(\rho V_s), \rho=2000.                                                                         |
| `dist_edge`     | Lateral distance [m] from the recorder to the nearest major impedance contrast (\lvert\mathrm{impgrad}\rvert above the 90th percentile) or strip edge. |
| `x_over_L`      | x / L with L = 500 m (variability-strip length). Mesh-agnostic lateral position.                                                                       |
| `x_over_lambda` | x / \lambda with \lambda = V_{s,\mathrm{col}} / f (column-mean soil Vs). Position in wavelengths.                                                      |




### Physics / OrbitAll equivalents


| Name                 | Definition                                                                                                                                                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `r_H`                | Horizontal correlation length from H5 `params` (sample-level).                                                                                                                                                                                                                 |
| `xi_k_re`, `xi_k_im` | Top-K=8 KL-equivalent **FFT spectral coefficients**, stored as **real/imag pairs** (16 scalars). Recovered by replaying `generate_gaussian_field_fft` white-noise with `rf_seed` and ranking modes by the exponential-covariance PSD (same generative model as data creation). |
| `f_star`             | Non-dimensional frequency f \cdot H / V_{s,\mathrm{col}} — teaches resonance scaling rather than raw Hz.                                                                                                                                                                       |
| `sin_f`, `cos_f`     | \sin(2\pi \hat f),\ \cos(2\pi \hat f) with \hat f log-scaled into [0,1] over 0.1–10 Hz (Fourier features for high-frequency extrapolation).                                                                                                                                    |


---



## 4. Screening protocol

1. **Stratified subsample** of n cases by CoV / H quantile bins (seed 42).
2. Cache full-frequency residuals at all **21** recorders.
3. Build a tabular design matrix: all recorders × **50** log-spaced frequency indices →
  - n=100 → 105 000 rows
  - n=1000 → **1 050 000** rows (`feature_table.parquet`)
4. **Mutual information** (`sklearn.feature_selection.mutual_info_regression`) vs each target, globally and in bands:
  low (0.1–0.5 Hz), mid (0.5–2 Hz), high (2–10 Hz).
5. **Random Forest** (200 trees) + **permutation importance** (\Delta R^2, 10 repeats) on a 50 000-row draw (train/test 75/25).

Entry point:

```bash
uv run python experiments/Residual/run_screen.py --n-samples 1000 --k-xi 8 --n-freq-screen 50
```

---



## 5. Quantitative results (n = 1000)



### 5.1 Predictive power of the feature set


| Target           | Train R^2 | Test R^2 | Gate hint                                   |
| ---------------- | --------- | -------- | ------------------------------------------- |
| R_{\mathrm{col}} | 0.73      | **0.40** | pass (features explain nontrivial variance) |
| R_{\mathrm{nom}} | 0.76      | **0.50** | pass                                        |


Pilot n=100 was higher (≈0.53 / 0.62 test R^2); the drop at n=1000 is expected (more diversity, same RF capacity / 50k-row subsample). The **ranking of features is stable**.

### 5.2 Agreed top features (MI ∩ permutation)

- **R_{\mathrm{col}}:** `f_star`, `x_over_lambda`, `sin_f`, `cos_f`
- **R_{\mathrm{nom}}:** `f_star`, `sin_f`, `cos_f`



### 5.3 Mutual information (band = all)


| Feature         | MI → R_{\mathrm{col}} | MI → R_{\mathrm{nom}} |
| --------------- | --------------------- | --------------------- |
| `f_star`        | **0.79**              | **0.87**              |
| `x_over_lambda` | 0.34                  | 0.34                  |
| `sin_f`         | 0.34                  | 0.35                  |
| `cos_f`         | 0.09                  | 0.11                  |
| `r_H`           | 0.05                  | 0.07                  |
| `imp_grad`      | 0.01                  | 0.01                  |
| `dist_edge`     | ≈0                    | ≈0                    |
| `x_over_L`      | ≈0                    | ≈0                    |
| `dip_slope`     | **0**                 | **0**                 |


Low-frequency band (0.1–0.5 Hz) elevates `r_H` and several `xi_*` MI scores; high band is still dominated by `f_star`.

### 5.4 RF permutation importance (test \Delta R^2)

**R_{\mathrm{col}}** — top drivers: `f_star` (≈1.10), then `cos_f`, selected `xi_`* re/im pairs, `sin_f`, `x_over_lambda`. Geometry (`imp_grad`, `dist_edge`) is small but nonzero. `dip_slope` is exactly zero.

**R_{\mathrm{nom}}** — same pattern; `f_star` again dominates (≈1.04). Spectral `xi_`* and `imp_grad` appear slightly more often in the RF top-8 than for R_{\mathrm{col}} (nominal residual still “sees” missing RF content).

### 5.5 Interpretation


| Finding                                           | Implication for a future GNO                                                                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `f_star` **+ Fourier freq features dominate**     | Resonance / frequency context must be explicit inputs; do not feed raw Hz alone.                                                             |
| `x_over_lambda` **beats** `x_over_L`              | Lateral position should be wavelength-normalized for extrapolation.                                                                          |
| `xi_k_re/im` **moderate,** `r_H` **weak in RF**   | Stochastic context is useful but secondary; keep real/imag KL-equivalent coeffs from `rf_seed`.                                              |
| `dip_slope` **dead**                              | Expected: generation used `INTERLAYER_AMPLITUDE=0`. Do not treat dip as a required channel on *this* corpus; re-test on dipping experiments. |
| `imp_grad` **/** `dist_edge` **weak**             | Lateral impedance geometry is not the main variance driver of these residuals under the current RF gate — frequency physics is.              |
| **R_{\mathrm{nom}} easier than R_{\mathrm{col}}** | Nominal residual includes 1D RF mismatch; column residual is the stricter lateral-coupling target for OrbitAll-style operators.              |


**Gate verdict:** Engineered features **do** explain substantial nonlinear variance (test R^2 \sim 0.4–0.5). Frequency nondimensionalization and Fourier features are **required** inputs for a downstream operator. Pure geometric dip is **not** informative on this flat-interface GIFNO set.

---



## 6. Diagnostic plot

`[results/r_central_3x3.png](results/r_central_3x3.png)` — 3×3 panel of |R|(x_{\mathrm{central}}, f) for 9 random subsample cases, overlaying R_{\mathrm{col}} and R_{\mathrm{nom}}.

Typical pattern: orange |R_{\mathrm{nom}}| peaks above blue |R_{\mathrm{col}}|, i.e. local-column Haskell removes a large share of the 1D amplification error; remaining blue peaks are the lateral-coupling residual of interest.

Importance bar charts: `importance_R_col.png`, `importance_R_nom.png`.

---



## 7. Artifact index

```
experiments/Residual/
  run_screen.py          # CLI
  config.py              # paths, K_XI=8, bands, …
  haskell_baseline.py    # TF_1D_col + TF_1D_nom
  features.py            # geometry + spectral KL real/imag
  residual_target.py     # residual cache builder
  build_table.py         # parquet feature table
  screen_mi.py / screen_rf.py / plots.py
  cache/n1000_seed42/    # r_col.npy, r_nom.npy, feature_table.parquet, meta
  cache/n100_seed42/     # pilot
  results/
    gate_summary.json
    mi_R_col.csv, mi_R_nom.csv
    rf_perm_R_col.csv, rf_perm_R_nom.csv
    rf_metrics_*.json
    r_central_3x3.png
    importance_R_*.png
    run_n1000.log
```

---



## 8. Next steps (out of scope here)

- Train an operator that predicts R_{\mathrm{col}} (or a signed residual) from the **kept** feature set (`f_star`, Fourier freq, `x_over_lambda`, `xi_`*, optionally `r_H` / weak geometry).
- Re-run the same gate on **dipping** datasets where \nabla_x Z \neq 0.
- Optional: full 7680 pass only after the operator architecture is chosen; the n=1000 gate is already decisive on feature rankings.

