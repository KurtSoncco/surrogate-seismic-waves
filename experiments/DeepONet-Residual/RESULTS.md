# DeepONet-Residual — Results

**Experiment:** [`experiments/DeepONet-Residual/`](.)  
**Sibling feature gate:** [`../Residual/RESULTS.md`](../Residual/RESULTS.md)  
**Primary data:** stratified **n = 1000** (seed 42), sample split 700 / 150 / 150  
**Focus target:** signed **\(R_{\mathrm{nom}}\)** (nominal Haskell baseline)

This folder trains a mesh-agnostic DeepONet for

\[
R_*(x,f) = TF_{2D}(x,f) - TF_{1D,*}(x,f),
\]

with reconstruction \(\widehat{TF}=TF_{1D}+\widehat{R}\). Success is judged on **residual** metrics (`r2_R`, `pearson_R_freq`, `delta_r2_TF`), not raw TF R² (a lazy \(\widehat{R}\approx 0\) already looks strong when TF₁D is good).

---

## 1. Architecture

Park et al. (2026): **single shared branch** for tightly coupled material fields.

| Module | Design |
|--------|--------|
| **Trunk** | MLP on \((f^*,\sin,\cos,x/\lambda)\) → basis \(b\in\mathbb{R}^{128}\) |
| **Branch fusion** | Field encoding + stochastic \(\xi_{1..8}^{\mathrm{re/im}}, r_H, a_{HV}, \mathrm{CoV}, \xi_{\mathrm{damp}}\) → coefficients \(p\) |
| **Combine** | \(\widehat{R}=\sum_k p_k b_k + \beta\) |

### Field encoders compared (this update)

Both take stacked recorder-column fields \((V_s, \zeta, Z=\rho V_s)\) shaped `(B, 3, 128, 21)`.

| Encoder | Structure | Role |
|---------|-----------|------|
| **Conv** (baseline) | 2× Conv2d + pool + AdaptiveAvgPool → vector | Shallow global summary |
| **ResUNet** | Stem ResBlock → 2× Down (Res) → bottleneck → 2× Up with skips → AdaptiveAvgPool → vector | Multi-scale residual features with U-Net skips |

Training: **SmoothL1 only**, **AdamW** `betas=(0.9, 0.999)`, `lr=1e-3`, patience 60.

Reproduce encoder compare:

```bash
uv run python experiments/DeepONet-Residual/compare_encoders.py \
  --cache-tag n1000_seed42 --epochs 300 --patience 60
```

---

## 2. Why \(R_{\mathrm{nom}}\) (not \(R_{\mathrm{col}}\))

| Target | TF₁D-only R² | Residual learning incentive |
|--------|--------------|-----------------------------|
| \(R_{\mathrm{col}}\) | ~0.62 | Small — column Haskell already strong |
| **\(R_{\mathrm{nom}}\)** | ~0.47 | **Large** — RF + interlayer + 2D must be learned |

On n=1000 single-branch Conv (full trunk):

| Target | R²(R) | Pearson_R_freq | ΔR²_TF |
|--------|-------|----------------|--------|
| \(R_{\mathrm{col}}\) | 0.084 | 0.362 | +0.033 |
| **\(R_{\mathrm{nom}}\)** | **0.242** | **0.542** | **+0.131** |

Further work focuses on **\(R_{\mathrm{nom}}\)**.

---

## 3. Conv vs ResUNet on \(R_{\mathrm{nom}}\) (n=1000 test)

Same split, trunk, stochastic features, loss, and optimizer. Conv checkpoint reused from the prior long run; ResUNet trained to early stop (~88 epochs).

### Aggregate test metrics

| Encoder | R²(R) | Pearson_R | Pearson_R_freq | ΔR²_TF | rel-L2(R) | Pearson_TF_freq |
|---------|-------|-----------|----------------|--------|-----------|-----------------|
| Conv | 0.242 | 0.494 | 0.542 | +0.131 | 0.866 | 0.845 |
| **ResUNet** | **0.260** | **0.515** | **0.562** | **+0.141** | **0.855** | **0.853** |

ResUNet is a **consistent but moderate** upgrade on every residual-first score (+0.02 R²(R), +0.02 Pearson_R_freq, +0.01 ΔR²_TF). Neither reaches the RF-gate R² band 0.4–0.5 on **signed** \(R\) yet; Pearson-across-freq (~0.56) is already in that ballpark.

### Per-sample distributions (test, n=150)

| Metric | Conv median | ResUNet median |
|--------|-------------|----------------|
| R²(R) | 0.314 | 0.302 |
| Pearson_R_freq | 0.599 | **0.622** |
| ΔR²_TF | 0.127 | **0.135** |

Medians are close; ResUNet shifts the **upper half** of Pearson / ΔTF higher (see boxplots). Variance across samples remains large — some cases are still hard for both encoders.

### Plots

| Figure | File |
|--------|------|
| Metric boxplots | [`results/encoder_box_R_nom_n1000_seed42.png`](results/encoder_box_R_nom_n1000_seed42.png) |
| Metric histograms | [`results/encoder_hist_R_nom_n1000_seed42.png`](results/encoder_hist_R_nom_n1000_seed42.png) |
| TF₂D vs TF₁D vs TF₁D+R̂ | [`results/encoder_tf_R_nom_n1000_seed42.png`](results/encoder_tf_R_nom_n1000_seed42.png) |
| Signed residual spectra | [`results/encoder_residual_R_nom_n1000_seed42.png`](results/encoder_residual_R_nom_n1000_seed42.png) |
| JSON summary | [`results/encoder_compare_R_nom_n1000_seed42.json`](results/encoder_compare_R_nom_n1000_seed42.json) |

**TF curves (central recorder):** both encoders correct peak frequency relative to nominal TF₁D. ResUNet often produces **sharper / taller** peaks (better when Conv under-shoots; can overshoot amplitude on some samples). Residual spectra show ResUNet tracking true \(R_{\mathrm{nom}}\) slightly more closely near resonances.

---

## 4. Features that matter (unchanged lessons)

From Residual MI/RF gate + this operator:

| Keep in trunk | Keep in branch | Drop / low priority on flat GIFNO set |
|---------------|----------------|----------------------------------------|
| \(f^*\), Fourier \(\sin/\cos\) | \(\xi\) re/im (K=8), \(r_H\), CoV, \(\xi_{\mathrm{damp}}\) | `dip_slope` (dead), pure geometry |
| \(x/\lambda\) (beats \(x/L\)) | Field stack \(V_s,\zeta,Z\) via **ResUNet** (preferred) or Conv | Raw pixel \(x\), raw Hz alone |

Field encoder quality matters, but trunk frequency physics remains the primary inductive prior.

---

## 5. Metrics reminder (anti-lazy)

| Use | Avoid as sole success |
|-----|------------------------|
| `r2_R`, `pearson_R_freq`, `delta_r2_TF` | Raw `r2_TF` without `r2_TF_1d_only` |
| SmoothL1 on signed \(R\) | Claiming success when ΔR²_TF ≈ 0 |

TF₁D-only R² on this \(R_{\mathrm{nom}}\) test set is **0.474**; Conv / ResUNet raise reconstructed TF R² to **0.605 / 0.615** (Δ +0.13 / +0.14).

---

## 6. Checkpoints & reproduce

| Model | Checkpoint |
|-------|------------|
| Conv · \(R_{\mathrm{nom}}\) | `checkpoints/single_full_R_nom_n1000_seed42.pt` |
| ResUNet · \(R_{\mathrm{nom}}\) | `checkpoints/single_resunet_full_R_nom_n1000_seed42.pt` |

```bash
# Cache
uv run python experiments/DeepONet-Residual/residual_signed.py --cache-tag n1000_seed42

# Train ResUNet only (or full compare)
uv run python experiments/DeepONet-Residual/train.py \
  --cache-tag n1000_seed42 --target R_nom --field-encoder resunet \
  --epochs 300 --patience 60

uv run python experiments/DeepONet-Residual/compare_encoders.py \
  --cache-tag n1000_seed42 --skip-train   # plots/metrics only
```

---

## 7. Next steps

1. Keep **ResUNet + \(R_{\mathrm{nom}}\)** as the default encoder/target pair.
2. Close the gap to residual R² **0.4–0.5** (full-frequency eval, longer / no early stop, peak-band emphasis).
3. Optional: deeper ResUNet base width or attention at the bottleneck; still Single-branch fusion per Park et al.
4. GNO only if residual lateral coupling still needs message passing beyond a global branch vector.
