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

Superseded by **§9**. Ship **geometry-aware Haskell nom** + **mixed-domain train (P3)** + **serial TF₁D-conditioned ResUNet**. Park DeltaPhi retrieval and a Fourier residual on \(R\) unless a new domain re-opens a column-Haskell gap. Do not scale n or restore TH-FNO / GNO / LOGLO-POD heads for this leftover.

---

## 8. Scale-up + Box OOD (this round)

**Hypothesis.** Direct TF mapping fails OOD because it must invent layered/dipping physics. \(R_{\mathrm{nom}}\) can generalize if Thomson–Haskell already carries geometry and \(\hat R\) **fails soft** (\(\hat R\to 0\)) instead of producing garbage TF.

Canonical OOD is **Box** `$GIFNO_DATA_ROOT/ood_dipping` and `ood_three_layer` (not `~/seiskit/neural-operator/experiments/`).

### 8.1 OOD layout probe

On Kurt-Asus Box (`/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data`):

| Corpus | Size | H5 | TF cache | Root manifest |
|--------|------|----|----------|----------------|
| `ood_dipping/` | 619 MB | **960** `h5/run_*.h5` | none | yes (960 rows) |
| `ood_three_layer/` | 469 MB | **960** `h5/run_*.h5` | none | yes (960 rows) |
| IID `h5/` + `transfer_function/` | 1.6 GB + 1.7 GB | 7680 | `tf_per_sample.npy` (616 MB) | TF manifest 7680 |

**`ood_dipping` sample `run_0.h5`:** `Vs_realization_2D` shape **(80, 1500)**, 42 accel channels (21 base + 21 surface), `dt=0.01`. H5 `params` include `dip_angle_deg`, `dip_span`, `dip_direction`, `Vs1`, `Vs2`, `H` (not `H_discretized`), `CoV`, `rf_seed`. Haskell-**nom** uses attrs `(Vs1, H, Vs2)`. Haskell-**col** uses `soil_nz = H` on the cropped Vs column so local depth (live dip) is in the 1D stack.

**`ood_three_layer` sample `run_0.h5`:** shape **(32, 1500)**. H5 `params` have `Vs1`, `H1`/`H2`, `Vs_mid`, `Vs_bedrock`, `seed1`/`seed2` — **no** single-layer `(Vs1, H, Vs2)`. Haskell-**nom** is **misspecified by design**: top-layer `Vs1`, total soil \(H=H_1+H_2\), `Vs_bedrock` (not an equivalent 1-layer fit). Haskell-**col** uses `soil_nz` from rounded `H1+H2` on the actual 3-layer column.

OpenSees TF for OOD is computed from `recorders/accel` with channels **`[base | surface]`** (capability-check convention; these files have no `row_y_m`), Konno–Ohmachi on the GIFNO 0.1–10 Hz / 1000-bin grid, cached under `experiments/DeepONet-Residual/cache/ood_*_tf/`.

LOGLO-POD leftovers to beat on OOD (not the 16–26 IID figure): dipping rel L2 ≈ **0.77–0.79** (Pearson ≈ 0.47–0.53); three_layer case_0 rel L2 ≈ **0.97** (Pearson ≈ 0.09).

### 8.2 Infra

- Stratified CoV×H indices **without** Residual RF screen: `n1000 ⊂ n2000 ⊂ n3000` (seed 42) written under `cache/n*_seed42/sample_indices.npy`.
- `--n-freq-train` (default 50); reported test metrics always use **1000** frequency bins.
- `eval_ood.py` walks Box `ood_*` (defaults `$GIFNO_DATA_ROOT/ood_dipping` and `ood_three_layer`). `run_scale.py`: `cache_tag × encoder × n_freq × seed`.
- `stage_screen_pack.sh` copies TF cache + stratified H5 + OOD (do not commit H5).

### 8.3 E0 — OOD Haskell floor (no NN)

All **960 + 960** Box H5 files. Metrics vs OpenSees |TF|. LOGLO-POD leftovers: dipping rel L2 ≈ 0.77–0.79 (Pearson ≈ 0.47–0.53); three_layer case_0 rel L2 ≈ 0.97 (Pearson ≈ 0.09).

**`ood_dipping` (n=960).** Nom uses `(Vs1, H, Vs2)`; dip is live so col should beat nom.

| Baseline | rel L2 mean | Pearson_freq mean | R²(TF) mean |
|----------|-------------|-------------------|-------------|
| Haskell-nom | **0.578** | 0.635 | 0.268 |
| Haskell-col | **0.541** | **0.711** | **0.353** |
| LOGLO-POD (phase0c leftover) | 0.77–0.79 | ~0.47–0.53 | — |

**`ood_three_layer` (n=960).** Nom is misspecified (wrong layer count). Col is the 1D floor.

| Baseline | rel L2 mean | Pearson_freq mean | R²(TF) mean |
|----------|-------------|-------------------|-------------|
| Haskell-nom | **0.956** | ≈ 0 | −0.303 |
| Haskell-col | **0.592** | **0.765** | **0.499** |
| LOGLO-POD (phase0c leftover) | ~0.97 | ~0.09 | — |

**Go/no-go — OOD Haskell floor: PASS.** Column Haskell is well below the POD ~0.8–1.0 rel L2 band on both corpora. Dipping nom also beats POD. Three-layer nom matches POD’s failure (rel L2 0.96, no correlation): that gap **is** the \(R_{\mathrm{nom}}\) OOD question, not a reason to stop. Residual is still the right OOD story *if* \(\hat R\) can add missing resonances or fail soft to 0.

JSON: `results/ood_e0_haskell.json` (gitignored). Reproduce:

```bash
uv run python experiments/DeepONet-Residual/eval_ood.py \
  --out experiments/DeepONet-Residual/results/ood_e0_haskell.json
```

### 8.4 IID scale (E1–E3)

n=1000 ResUNet baseline to beat: `r2_R=0.260`, `delta_r2_TF=+0.141`, TF R² 0.615 vs Haskell-only 0.474.

Trained on RTX 5080 Laptop with **torch 2.11.0+cu128**. Test metrics below are always **1000 frequency bins**. E1 (`n_freq_train=50`) matched n=1000 `r2_R` and missed `Δr2_TF`, so **E3 (n=3000) was skipped**. E2 raised `r2_R` only ~0.01.

| n | n_freq_train | Encoder | r2_R | Pearson_R_freq | Δr2_TF | epochs | notes |
|---|--------------|---------|------|----------------|--------|--------|-------|
| 1000 | 50 | ResUNet | 0.260 | 0.562 | +0.141 | ~88 | existing (different 150-sample test split) |
| 2000 | 50 | ResUNet | 0.260 | 0.559 | +0.133 | 80 | E1 — **flat** |
| 2000 | 200 | ResUNet | **0.269** | **0.564** | +0.137 | 81 | E2 winner |
| 2000 | 1000 | ResUNet | 0.268 | 0.563 | +0.137 | 91 | E2 full-grid train |
| 3000 | — | — | — | — | — | — | E3 skipped (E1 flat) |

E4 checkpoint: `checkpoints/single_resunet_full_R_nom_n2000_seed42_nf200_seed42.pt`.

```bash
uv run python experiments/DeepONet-Residual/residual_signed.py --cache-tag n2000_seed42
uv run python experiments/DeepONet-Residual/run_scale.py \
  --cache-tag n2000_seed42 --encoder resunet --n-freq-train 50 --patience 60
# E2 (after E1 flat):
uv run python experiments/DeepONet-Residual/run_scale.py \
  --cache-tag n2000_seed42 --encoder resunet --n-freq-train 200 --patience 60
```

### 8.5 E4 — OOD residual vs Haskell floor

Winner recipe scored on all **960 + 960** Box H5s (`--clamp none`; tanh/zero also run). Metrics vs OpenSees |TF|.

**`ood_dipping` (n=960).** Same 1-layer nom as IID.

| Predictor | rel L2 mean | Pearson_freq mean | R²(TF) mean | beats nom | beats col |
|-----------|-------------|-------------------|-------------|-----------|-----------|
| Haskell-nom | 0.578 | 0.635 | 0.268 | — | — |
| Haskell-col | 0.541 | 0.711 | 0.353 | — | — |
| nom + \(\hat R\) (none) | **0.538** | **0.713** | 0.346 | 84% | 54% |
| nom + \(\hat R\) (tanh) | **0.538** | 0.691 | **0.357** | 96% | 56% |
| nom + \(\hat R=0\) | 0.578 | 0.635 | 0.268 | 0% | 30% |

**`ood_three_layer` (n=960).** Nom misspecified (wrong layer count).

| Predictor | rel L2 mean | Pearson_freq mean | R²(TF) mean | beats nom | beats col |
|-----------|-------------|-------------------|-------------|-----------|-----------|
| Haskell-nom | 0.956 | ≈ 0 | −0.303 | — | — |
| Haskell-col | **0.592** | **0.765** | **0.499** | — | — |
| nom + \(\hat R\) (none) | 0.938 | 0.078 | −0.255 | 74% | **0%** |
| nom + \(\hat R\) (tanh) | 0.945 | 0.040 | −0.273 | 73% | 0% |
| nom + \(\hat R=0\) | 0.956 | ≈ 0 | −0.303 | 0% | 0% |

Unclamped and tanh are tied on dipping rel L2 (0.538); tanh wins TF R² (0.357 vs 0.346) and file-wise beat rate. Neither closes the three-layer gap. Zero is Haskell-nom by construction.

JSON: `results/ood_e4.json`, `ood_e4_clamp_tanh.json`, `ood_e4_clamp_zero.json` (gitignored).

```bash
uv run python experiments/DeepONet-Residual/eval_ood.py \
  --checkpoint experiments/DeepONet-Residual/checkpoints/single_resunet_full_R_nom_n2000_seed42_nf200_seed42.pt \
  --out experiments/DeepONet-Residual/results/ood_e4.json
```

### 8.6 Verdict (R_nom generalization)

**Not feasible is ruled out at the Haskell-floor gate** (E0). Column (and dipping nom) 1D physics already beats the in-repo LOGLO-POD OOD leftovers.

IID n=2000 does **not** scale residual skill past n=1000 (`r2_R` 0.260 → 0.269; `Δr2_TF` +0.141 → +0.137). Extra trunk frequencies do not change that.

OOD label is **split by corpus**, not a single **feasible** / **IID-only** stamp:

- **`ood_dipping` — feasible.** `TF_{1D,nom}+\hat R` beats Haskell-nom (rel L2 0.538 vs 0.578) and matches Haskell-col (0.538 vs 0.541). The residual adds the live-dip / RF leftover that nom misses, and fails soft enough not to trash the 1D floor.
- **`ood_three_layer` — IID-only against a 1-layer nom.** \(\hat R\) slightly improves misspecified nom (0.938 vs 0.956) but **never** beats column Haskell (0.592) and Pearson stays ~0.08. The gap is wrong layer count (see §8.7).

So \(R_{\mathrm{nom}}\) is a dipping/RF correction, not a substitute for a correctly specified 1D stack.

### 8.7 True 3-layer Haskell nom (no retraining)

Cheap check: score `ood_three_layer` with Thomson–Haskell on attrs \((V_{s1}, H_1, V_{s,\mathrm{mid}}, H_2, V_{s,\mathrm{bedrock}})\) instead of the misspecified 1-layer \((V_{s1}, H_1+H_2, \mathrm{bedrock})\). Same 960 H5s, same TF cache, same E2 checkpoint (IID \(\hat R\) trained vs 1-layer nom).

| Predictor | rel L2 mean | Pearson_freq mean | R²(TF) mean | beats 1-layer nom | beats col |
|-----------|-------------|-------------------|-------------|-------------------|-----------|
| Haskell-nom (1-layer, misspecified) | 0.956 | ≈ 0 | −0.303 | — | — |
| Haskell-nom3 (true 3-layer) | **0.683** | **0.667** | **0.333** | **99.6%** | 12% |
| nom3 + IID \(\hat R\) | 0.673 | 0.684 | 0.350 | — | 16% |
| Haskell-col | **0.592** | **0.765** | **0.499** | — | — |

**Outcome.** Correct layering is most of the three-layer OOD story: Pearson 0 → 0.67 and rel L2 0.96 → 0.68, beating misspecified nom on essentially every file. A 2D/RF leftover remains vs column Haskell (0.68 vs 0.59). Sticking the IID 1-layer \(\hat R\) on top of nom3 helps only ~0.01 rel L2 — the residual was trained to correct the wrong 1D operator.

Generalization recipe: **call the right 1D stack at test time** (1-layer nom on dipping / IID, 3-layer nom here). Train a new \(R\) against nom3 only if you need to close that last ~0.09 rel L2, and then mix 3-layer geometry into training. Do not scale n or change the encoder for this gap.

```bash
uv run python experiments/DeepONet-Residual/eval_ood.py \
  --corpus ood_three_layer \
  --checkpoint experiments/DeepONet-Residual/checkpoints/single_resunet_full_R_nom_n2000_seed42_nf200_seed42.pt \
  --out experiments/DeepONet-Residual/results/ood_e4_nom3.json
```

JSON: `results/ood_e4_nom3.json` (gitignored).

---

## 9. Operator, domain mix, and architecture (this round)

IID is a **1-soil + bedrock** stack. Box `ood_dipping` keeps that nom and adds live dip. Box `ood_three_layer` is **2-soil + bedrock**. Each Box corpus is split **70/15/15, seed 42** (672 / 144 / 144) so mix / sequential training is a real experiment; held-out slices are the generalization test. IID uses the nested `n1000_seed42` split (700 / 150 / 150). Geometry-aware signed \(R\) is the training target: 1-layer nom on IID/dipping, **true 3-layer nom** on three_layer. All nets: ResUNet unless noted, `n_freq_train=200`, 1000-bin eval, SmoothL1, AdamW, patience 60 (P2 finetune lr \(10^{-4}\), patience 30). RTX 5080 Laptop.

Reproduce (caches already on disk):

```bash
GIFNO_DATA_ROOT=data/gifno_screen \
GIFNO_OOD_DIPPING=data/gifno_screen/ood_dipping \
GIFNO_OOD_THREE_LAYER=data/gifno_screen/ood_three_layer \
uv run python experiments/DeepONet-Residual/domain_study.py --skip-cache
```

JSON: `results/domain_study/{operator_bakeoff,protocols,architectures,summary}.json`.

### 9.1 Which 1D operator (no new net)

Held-out test slices vs OpenSees |TF|. `nom*` = geometry-aware prior. Frozen E2 is the n=2000 ResUNet \(\hat R\) from §8, scored on these **test** slices (not the old full-960 numbers).

| Prior | IID test (n=150) rel L2 / Pearson_freq | dipping test (n=144) | 3-layer test (n=144) |
|-------|------------------------------------------|----------------------|----------------------|
| Haskell-nom (1-layer) | 0.565 / 0.745 | 0.601 / 0.630 | **misspecified** 0.959 / 0.012 |
| Haskell-nom3 (true stack) | — | — | **0.730 / 0.665** |
| Haskell-col | **0.489 / 0.841** | **0.570 / 0.702** | **0.635 / 0.770** |
| nom* + frozen E2 \(\hat R\) | 0.479 / 0.853 | 0.571 / 0.700 | 0.716 / 0.685 |

**Go:** geometry-aware nom is the default training target. Column Haskell remains the 1D ceiling. Misspecified 1-layer nom on three_layer is only an ablation (Pearson ≈ 0). Frozen IID \(\hat R\) beats nom on dipping (0.571 vs 0.601) and matches col, but barely moves nom3 on three_layer (0.716 vs 0.730) — same story as §8.7 on a fair held-out slice.

### 9.2 Domain protocols (same ResUNet, geometry-aware \(R_{\mathrm{nom}}\))

Winner rule: best three_layer test **rel L2** among protocols that keep IID `r2_R ≥ −0.05`.

| ID | Train | IID `r2_R` / rel L2 / Δr²_TF | dipping | 3-layer | vs col (3L 0.635) |
|----|--------|------------------------------|---------|---------|-------------------|
| P0 | IID n=1000 only | **0.213** / 0.499 / +0.124 | 0.038 / 0.574 / +0.067 | 0.004 / 0.723 / +0.014 | no transfer |
| P1 | IID + dipping | 0.199 / 0.503 / +0.116 | **0.571** / **0.383** / +0.439 | −0.008 / 0.727 / +0.005 | dip leftover learned; 3L still IID-only |
| P2 | P1 → finetune 3-layer | **−18.8** / 2.50 / −10.4 | −1.86 / 0.989 / −1.26 | 0.203 / 0.646 / +0.154 | 3L almost col; **catastrophic forgetting** |
| **P3** | **mix all three** | **0.217** / 0.497 / +0.126 | **0.569** / **0.384** / +0.437 | **0.207** / **0.645** / +0.156 | **winner** — 3L without collapsing IID/dip |
| P4 | 3-layer only | −7.09 / 1.60 / −3.93 | −3.72 / 1.27 / −2.56 | **0.237** / **0.632** / +0.178 | best 3L, reverse-transfer fails |

P1 shows dip is **in-family**: once dipping train is seen, \(\hat R\) crushes the live-dip leftover (rel L2 0.383 vs col 0.570) with almost no IID cost. Extra layers are **not** in-family: P0/P1 do not transfer to three_layer, and P2/P4 learn the 3-layer residual only by destroying IID/dipping. **Mix (P3)** is the protocol that should ship: IID matches P0, dipping matches P1, three_layer matches P2's 3-layer skill without the forgetting.

### 9.3 Architecture on P3 mix

Operator fixed (geometry-aware nom). ResUNet numbers are P3 from §9.2 (not retrained).

| Arch | IID `r2_R` / rel L2 / Pearson_R_freq | dipping | 3-layer | vs Haskell-col |
|------|--------------------------------------|---------|---------|----------------|
| Conv DeepONet | 0.236 / 0.491 / 0.546 | 0.579 / 0.379 / 0.772 | **0.246 / 0.629 / 0.553** | **beats col on 3L** (0.629 vs 0.635) |
| ResUNet DeepONet (P3) | 0.217 / 0.497 / 0.524 | 0.569 / 0.384 / 0.767 | 0.207 / 0.645 / 0.508 | 3L ≈ col |
| MIONet-style multi | 0.199 / 0.503 / 0.532 | 0.562 / 0.387 / 0.763 | 0.225 / 0.637 / 0.533 | slightly worse than single-branch |
| **Serial TF₁D-conditioned** | **0.437 / 0.422 / 0.707** | **0.597 / 0.371 / 0.791** | 0.234 / 0.634 / 0.540 | **beats col on IID** (0.422 vs 0.489) and dipping; 3L ties col |

Conditioning \(\hat R\) on \(\log\mathrm{TF}_{1D}\) (serial / discrepancy operator) is the architecture win: IID residual R² jumps 0.22 → **0.44**, TF rel L2 **0.422** vs column Haskell **0.489**. Conv is the 3-layer rel-L2 winner by a hair (0.629). Multi-branch is not better when layer count changes — Park et al.'s single-branch preference holds on this mix.

### 9.4 Five literature ideas (what we did)

1. **Geometry-aware prior + additive correction** ([arXiv:2606.03469](https://arxiv.org/html/2606.03469)) — implemented as the default operator (A + B). Switching the Haskell stack with domain is most of the 3-layer story; the net only learns leftover.
2. **Serial / discrepancy operator** (DeepFNOnet [arXiv:2502.11279](https://arxiv.org/html/2502.11279), serial DeepONet in 2606.03469) — implemented as architecture 4. **Best overall** on mix.
3. **MIONet multiple-input product** ([arXiv:2202.06137](https://arxiv.org/abs/2202.06137)) — implemented as `multi`. Slightly worse than single-branch on every domain.
4. **DeltaPhi retrieve-and-residual** ([arXiv:2406.09795](https://arxiv.org/abs/2406.09795) / NeurIPS 2025) — **not trained**. Would retrieve a nearby column-Haskell or IID neighbor at inference and learn the residual to that state, not to a global nom. Needs a retrieval index; parked.
5. **Multiscale / Fourier residual on \(R(x,f)\)** (Multiscale DeepONet [arXiv:2111.04860](https://arxiv.org/abs/2111.04860), Fourier-MIONet [arXiv:2303.04778](https://arxiv.org/abs/2303.04778)) — **stretch, not run**. Serial + mix already closed the column-Haskell gap on every held-out slice. Revisit only if a new geometry re-opens it.

Out of scope (unchanged): PI-DeepONet PDE residuals, DeltaPhi. GNO / FNO-on-\(R\) / n-ladder: **§10**.

### 9.5 Verdict

- **1D prior to ship:** geometry-aware Thomson–Haskell (1-layer nom on IID/dipping, true 3-layer nom on `ood_three_layer`). Column Haskell is the ceiling, not the training target.
- **Mix, not sequential.** P3 keeps IID and dipping while learning the 3-layer leftover. P2 sequential 2-layer → 3-layer **forgets**. P4 3-layer-only does not reverse-transfer.
- **Architecture to ship:** serial TF₁D-conditioned single-branch ResUNet on the P3 mix. Conv is a cheaper alternative if you only score 3-layer rel L2.
- **vs GIFNO-XT:** residual loses IID (0.422 vs 0.302) and **wins OOD** (dipping 0.371 vs ~0.8 / 16–26; three_layer 0.634 vs ~0.97 / 16–26). That is the intended split.
- **Do not train DeltaPhi or Fourier-\(R\) this round** — the col gap is closed on these test slices.

Checkpoint: `checkpoints/arch_serial_P3_mix.pt`.

### 9.6 vs GIFNO-XT (no residual)

Direct TF mapping is LOGLO-POD / GIFNO-FDO-XT (`test_rel_l2` **0.302** on the 2000-sample IID screen; Pearson **0.919**). That model **does not** carry a 1D prior: OOD capability checks on seiskit 3-layer / dipping collapse (rel L2 **~16–26**, Pearson ≈ 0). On Box OOD leftovers cited in §8, dipping is still ~**0.77–0.79** and three_layer case_0 ~**0.97**.

Same-space comparison for the shipped serial P3 mix (held-out test slices, linear |TF|):

| Domain | GIFNO-XT rel L2 / Pearson | Haskell-nom* rel L2 / Pearson_freq | **Shipped residual** rel L2 / Pearson_freq | Winner |
|--------|---------------------------|------------------------------------|--------------------------------------------|--------|
| IID | **0.302 / 0.919** (n=2000 screen; per-recorder Pearson 0.939) | 0.565 / 0.745 | 0.422 / **0.900** (n=150 nested test) | **GIFNO-XT on L2**; Pearson almost tied |
| dipping | ~0.77–0.79 / ~0.47–0.53 (Box leftover); 16–26 / ≈0 (seiskit) | 0.601 / 0.630 | **0.371 / 0.880** | **residual** |
| three_layer | ~0.97 / ~0.09 (case_0); 16–26 / ≈0 (seiskit) | 0.730 / 0.665 (nom3) | **0.634 / 0.802** | **residual** |

GIFNO-XT `test_pearson` is a global pool; residual `pearson_TF_freq` is mean per-recorder spectrum correlation (closest to GIFNO-XT `test_pearson_mean` **0.939**). On IID the residual is **behind on amplitude L2** but **nearly tied on spectral shape**. On OOD, Pearson is the clearer win: GIFNO-XT leftover spectra are weakly correlated or uncorrelated; nom*+\(\hat R\) keeps the resonances.

The residual is **not** trying to beat GIFNO-XT on in-family IID amplitude. It exists because GIFNO-XT has to invent layered/dipping physics from a 1-soil training prior and fails OOD. On that job the shipped operator wins by a wide margin (and matches or beats column Haskell). IID Pearson is already close (0.90 vs 0.92–0.94); the leftover gap is L2, not shape. No DeltaPhi / Fourier-\(R\) follow-up: OOD vs the no-residual baseline is already good.

Follow-up (n-ladder, train recipe, FNO-on-\(R\), recorder GNO / GINO): **§10**. The GINO residual now ships (`checkpoints/M700_gino.pt`).

---

## 10. Training recipe, n-ladder, FNO-on-\(R\), and recorder GNO (this round)

Control is the §9 serial ResUNet P3 mix (`arch_serial_P3_mix.pt`): SmoothL1, `n_freq_train=200`, 1000-bin eval, patience 60, RTX 5080 Laptop. Held-out tests stay seed-42 (IID 150, dip 144, 3L 144). Extra IID for M1400 is taken from n2000 samples **outside** the n1000 corpus — naive `make_splits(2000)` would leak 107 of 150 n1000 test files. Logs: wandb project `deeponet-residual` (offline on this laptop) + tqdm epoch/batch bars. Reproduce:

```bash
GIFNO_DATA_ROOT=data/gifno_screen \
GIFNO_OOD_DIPPING=data/gifno_screen/ood_dipping \
GIFNO_OOD_THREE_LAYER=data/gifno_screen/ood_three_layer \
uv run python experiments/DeepONet-Residual/arch_train.py --mix M1400 --run-name M1400_serial
uv run python experiments/DeepONet-Residual/arch_train.py --mix M700 --encoder gno --fno --run-name M700_gino
uv run python experiments/DeepONet-Residual/eval_ood.py --checkpoint experiments/DeepONet-Residual/checkpoints/M700_gino.pt --split test
```

**Gates:** IID rel L2 toward GIFNO-XT **0.30** without collapsing OOD (dipping ≲ 0.37 / Pearson ≲ 0.88); three-layer **beat column Haskell** (per-file mean col **0.597**, pooled col **0.635**).

### 10.1 VRAM

| Setup | peak GB (bs=16) |
|-------|-----------------|
| serial ResUNet | 1.14 (bs=8: 0.60) |
| + FNO-on-\(R\) (32 ch, modes 8×16, 4 layers) | 1.23 |
| recorder GNO | 0.94 |

16 GB is not the limit; CPU preload is. Train batch stays **8** to match the shipped control. M7680 / Lambda not used: n-ladder was flat and FNO/GNO fit.

### 10.2 n-ladder (same serial ResUNet)

| Mix | IID train | + dip/3L | IID rel L2 / Pearson | dipping | 3-layer |
|-----|-----------|----------|----------------------|---------|---------|
| **M700** (control) | 700 | 672+672 | **0.422 / 0.900** | **0.371 / 0.880** | **0.634 / 0.802** |
| M1400 | 700+700 extra | 672+672 | 0.421 / 0.902 | 0.378 / 0.875 | 0.641 / 0.808 |

IID rel L2 drop 0.001 ≪ 0.02. Matches E1: more IID does not buy residual skill. **Stop** (no M2100, no Lambda M7680). Winning \(n\) = **M700**.

### 10.3 Train recipe (on M700)

| Recipe | IID rel L2 / Pearson | dipping | 3-layer |
|--------|----------------------|---------|---------|
| SmoothL1-on-\(R\) (control) | **0.422 / 0.900** | **0.371 / 0.880** | 0.634 / 0.802 |
| 50% IID resampling | 0.427 / 0.898 | 0.392 / 0.863 | 0.639 / 0.802 |
| aux TF rel L2 0.25 + 0.5–2 Hz peak 0.1 | 0.423 / 0.897 | 0.378 / 0.873 | **0.630 / 0.804** |

Resampling hurts. Aux TF L2 is a hair better on 3-layer and fails the IID ≤ 0.422 gate by 0.001. Keep **SmoothL1-only** as the training loss.

### 10.4 Architecture (M700, SmoothL1, serial \(\log\mathrm{TF}_{1D}\))

FNO is a 4-layer FNOBlocks residual on the \((21, n_f)\) \(\hat R\) grid (GELU, modes 8×16). GNO is per-recorder 1D depth conv + 3-layer chain message passing (kNN=2 along \(x\)), then a per-node DeepONet branch. GINO = GNO encoder + FNO-on-\(R\).

Pooled held-out test (rel L2 / Pearson_TF_freq):

| Arch | IID | dipping | 3-layer | vs col (pooled 0.489 / 0.570 / 0.635) |
|------|-----|---------|---------|----------------------------------------|
| serial ResUNet (control) | 0.422 / 0.900 | 0.371 / 0.880 | 0.634 / 0.802 | beats IID+dip; 3L ties |
| FNO-on-\(R\) | **0.371 / 0.918** | 0.345 / 0.892 | 0.615 / 0.812 | beats col on all three |
| recorder GNO | 0.420 / 0.899 | 0.372 / 0.873 | 0.550 / **0.876** | 3L is the GNO story |
| **GINO (GNO+FNO)** | **0.371 / 0.915** | **0.335 / 0.896** | **0.533 / 0.866** | **winner** |

GNO is the lateral leftover the global AdaptiveAvgPool was missing (Residual RF gate, §5.5). FNO is the oscillatory leftover vs column Haskell. Combining them does not trade one for the other.

`eval_ood --split test` on GINO (per-file mean, n=144):

| Corpus | Haskell-col mean rel L2 | GINO mean rel L2 / Pearson_freq | frac beats col |
|--------|-------------------------|---------------------------------|----------------|
| dipping | 0.547 | **0.312 / 0.896** | **99.3%** (serial was 98%) |
| three_layer | 0.597 | **0.481 / 0.866** | **82.6%** (serial was 49%) |

### 10.5 vs GIFNO-XT and ship

| Domain | GIFNO-XT | Haskell-col | serial P3 mix | **GINO** |
|--------|----------|-------------|---------------|----------|
| IID | **0.302 / 0.919** | 0.489 / 0.841 | 0.422 / 0.900 | **0.371 / 0.915** |
| dipping | ~0.77–0.79 | 0.570 / 0.702 | 0.371 / 0.880 | **0.335 / 0.896** |
| three_layer | ~0.97 | 0.635 / 0.770 | 0.634 / 0.802 | **0.533 / 0.866** |

IID Pearson is now tied with GIFNO-XT (0.915 vs 0.919). Amplitude L2 is still behind (0.371 vs 0.302). Three-layer **clears** column Haskell on pooled L2 and on 83% of held-out files.

**Ship:** `checkpoints/M700_gino.pt`. Keep `arch_serial_P3_mix.pt` as the §9 control. Skip mesh Transolver / DeltaPhi. GINO-wide + n-scale (did not unseat this ckpt): **§11**.

---

## 11. GINO-wide recipe and n-scale (this round)

Laptop §10 GINO is the control: width 32, modes \(8\times16\), `batch_size=8`, `n_freq_train=200` (IID **0.371 / 0.915**, dipping **0.335 / 0.896**, 3-layer **0.533 / 0.866**). This round widens FNO (width 64, modes \(8\times32\), 4 layers, `batch_size=32`) and n-scales that recipe only while IID rel L2 drops \(\ge 0.02\) with OOD held. Held-out slices stay seed-42 (IID 150, dip 144, 3L 144). SmoothL1 only. No TH-FNO / LOGLO-POD / GIFNO-XT / mesh Transolver.

**Host:** Lambda Labs `gpu_1x_a10` (`ubuntu@163.192.40.133`, NVIDIA A10 23 GB). Same files via rsync (`data/gifno_screen` + signed caches). Wandb project [`deeponet-residual`](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual) is **online** (`WANDB_MODE=online`, `host=lambda`). Laptop §10 + rehearsal runs were `wandb sync`'d from `experiments/DeepONet-Residual/wandb/offline-run-*`. VRAM of the wide recipe: **2.46 GB peak** on the 5080 (2.84M params).

```bash
bash experiments/DeepONet-Residual/lambda_train.sh \
  --mix M700 --encoder gno --fno --batch-size 32 \
  --fno-width 64 --fno-modes 8,32 --fno-layers 4 \
  --run-name M700_gino_wide_lambda
```

### 11.1 Wider GINO on M700 (same files)

| Recipe | host | IID rel L2 / Pearson | dipping | 3-layer | epochs | wandb |
|--------|------|----------------------|---------|---------|--------|-------|
| GINO M700 (control, §10) | 5080 | **0.371 / 0.915** | **0.335 / 0.896** | **0.533 / 0.866** | 96 | [synced](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual) |
| GINO-wide M700 | 5080 | 0.364 / 0.920 | 0.334 / 0.900 | 0.549 / 0.855 | 93 | [4vhdsnlf](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/4vhdsnlf) |
| GINO-wide M700 | **A10** | 0.376 / 0.915 | 0.340 / 0.896 | 0.538 / 0.861 | 113 | [zq6qxief](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/zq6qxief) |

A10 IID rel L2 **rose** 0.005 vs control (5080 rehearsal dropped 0.007). Neither hits \(\ge 0.02\). Dipping holds (\(\le 0.35\)). Three-layer is slightly worse. **Skip** `n_freq_train=1000`.

### 11.2 n-scale the wide recipe

| Mix | host | IID rel L2 / Pearson | dipping | 3-layer | ΔIID vs M700-wide | wandb |
|-----|------|----------------------|---------|---------|-------------------|-------|
| M700-wide | A10 | 0.376 / 0.915 | 0.340 / 0.896 | 0.538 / 0.861 | — | [zq6qxief](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/zq6qxief) |
| **M1400-wide** | **A10** | **0.343 / 0.929** | **0.328 / 0.900** | 0.542 / 0.855 | **−0.033** | [h94bwcl7](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/h94bwcl7) |
| M1400-wide | 5080 | 0.343 / 0.928 | 0.327 / 0.902 | 0.545 / 0.853 | −0.021 vs 5080-wide | [jg3u3s7f](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/jg3u3s7f) |

Unlike the serial ResUNet n-ladder (§10.2, ΔIID \(0.001\)), extra nested-safe IID **does** buy leftover L2 once FNO has width. OOD holds (dipping improved; 3-layer flat). Gate to M2100 **fires**.

**Stop before M2100 / M7680:** `cache/n3000_seed42` has indices only. The screen pack on the A10 still has **133 / 3000** H5 files; Box is not mounted. Nested-safe M7680 would also need a `n7680_seed42` Haskell pass.

### 11.3 Winner / ship

Winner = best 3-layer rel L2 among runs with IID \(\le 0.371\) and dipping \(\le 0.35\). Lambda M700-wide (IID 0.376) is out. Qualified: laptop GINO **0.533**, 5080 M1400-wide 0.545, A10 M1400-wide 0.542. **Laptop GINO 3-layer 0.533** still wins.

IID moved toward GIFNO-XT **0.30** (0.371 → **0.343**) without collapsing OOD, but the ship rule is three-layer leftover vs column Haskell, not IID L2.

**Ship:** keep [`checkpoints/M700_gino.pt`](checkpoints/M700_gino.pt). A10 ablations pulled to `checkpoints/M700_gino_wide_lambda.pt` and `M1400_gino_wide_lambda.pt`. Skip mesh Transolver / DeltaPhi / GIFNO-XT retrain.

### 11.4 Longer train + ReduceLROnPlateau (A10)

Fixed `lr=1e-3` was still in the weights at early stop. New loop: **ReduceLROnPlateau** on val SmoothL1 (`factor=0.5`, plateau 20, `min_lr=1e-6`), max **500** epochs, early-stop patience **80**, restore the **best val** checkpoint.

| Mix | best@ | stop | IID | dipping | 3-layer | wandb |
|-----|-------|------|-----|---------|---------|-------|
| M700-wide + sched | 26 | 106 | **0.357 / 0.924** | **0.334 / 0.900** | 0.552 / 0.855 | [o0ueo68d](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/o0ueo68d) |
| M1400-wide + sched | 36 | 116 | 0.349 / 0.928 | 0.333 / 0.899 | 0.545 / 0.854 | [8jrrje75](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/8jrrje75) |

LR dropped three times on each run (1e-3 → 1.25e-4); those later epochs did **not** beat the early val best, so the restored weights are the epoch-26/36 snapshots. M700-wide IID **0.357** is the first wide run to beat control 0.371 (Δ **−0.014**, still short of the 0.02 nf=1000 gate). M1400 vs that M700 is only **−0.008**, so n-scale does **not** fire under this recipe. Three-layer is still worse than laptop GINO **0.533**. **Ship unchanged.**

---

## 12. SOTA operator heads on the leftover grid (A10)

Same M700 mix, serial \(\log\mathrm{TF}_{1D}\), SmoothL1, ReduceLROnPlateau, bs=32, FNO width 64 / modes \(8\times32\) unless noted. Held-out seed-42 tests. Direct TF SOTA remains GIFNO-XT (IID **0.302 / 0.919**) which still collapses OOD.

Three published operator families, adapted to the 21-recorder × frequency leftover (not a full PDE mesh):

| Head | Paper | What we implemented |
|------|-------|---------------------|
| GINO (control) | Li et al. GINO / DeepFNOnet | chain GNO (kNN=2) + vanilla FNOBlocks on \(R\) |
| U-FNO | Wen et al. 2022 | FNO layer + local 3×3 conv residual each block |
| F-FNO | Tran et al. 2023 | factorized 1D spectral conv along recorder and freq |
| GNOT / Transolver-lite | Hao et al. GNOT; Wu et al. Transolver 2024 | self-attention over the 21 stations, then vanilla FNO-on-\(R\) |
| AFNO | Guibas et al. 2022 | shared channel MLP in Fourier space |
| WNO | Tripura & Chakraborty 2023 | 1-level Haar DWT on frequency + local conv |
| FNO-1D | Li et al. (1D spectral) | FNO along frequency only (GNO already mixes \(x\)) |
| GAT | Veličković et al. 2018 | local attention on {left, self, right}, then vanilla FNO |

Pooled held-out rel L2 / Pearson_TF_freq:

| Model | IID | dipping | 3-layer | vs col (0.489 / 0.570 / 0.635) | wandb |
|-------|-----|---------|---------|--------------------------------|-------|
| GIFNO-XT (no residual) | **0.302 / 0.919** | ~0.77–0.79 | ~0.97 | loses OOD | — |
| **GINO M700 (ship)** | 0.371 / 0.915 | **0.335 / 0.896** | **0.533 / 0.866** | **winner** | §10 |
| GINO-wide + sched | **0.357 / 0.924** | 0.334 / 0.900 | 0.552 / 0.855 | best IID among residuals | [o0ueo68d](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/o0ueo68d) |
| GAT + FNO | 0.364 / 0.920 | **0.331 / 0.902** | 0.542 / 0.860 | closest runner-up | [we6br04l](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/we6br04l) |
| U-FNO + GNO | 0.377 / 0.911 | 0.345 / 0.891 | 0.565 / 0.850 | local conv does not help \(R\) | [5koj3fcv](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/5koj3fcv) |
| Attn + FNO | 0.375 / 0.917 | 0.339 / 0.898 | 0.621 / 0.806 | 3L ≈ serial ResUNet | [4ylwfy6d](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/4ylwfy6d) |
| AFNO + GNO | 0.378 / 0.915 | 0.354 / 0.887 | 0.561 / 0.851 | dip misses 0.35 | [8rn3pvrn](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/8rn3pvrn) |
| FNO-1D + GNO | 0.395 / 0.911 | 0.372 / 0.871 | 0.565 / 0.846 | freq-only spectra too weak | [cljoe5ho](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/cljoe5ho) |
| F-FNO + GNO | 0.397 / 0.905 | 0.375 / 0.870 | 0.561 / 0.847 | dip fails 0.35 gate | [f997t5ua](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/f997t5ua) |
| WNO + GNO | 0.554 / 0.742 | 0.498 / 0.735 | 0.673 / 0.761 | Haar leftover is a miss | [7o0qq4e7](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/7o0qq4e7) |

**What this says.** The leftover is not a generic 2D PDE field. Extra local conv (U-FNO), factorized/1D/adaptive Fourier (F-FNO, FNO-1D, AFNO), and Haar wavelets all **hurt** vs vanilla FNO on \((21, n_f)\). Dense station attention **drops three-layer** back toward the pooled-encoder regime (0.621 vs GNO 0.533). **Local GAT** keeps the kNN=2 graph and is the only runner-up (dip 0.331, IID 0.364) but still loses three-layer to chain GNO (0.542 vs **0.533**). The line-graph GNO + vanilla FNO inductive bias is the one that fits.

None of the SOTA heads beat laptop GINO on three-layer, and none close GIFNO-XT’s IID L2 without giving up OOD. **Ship stays** [`checkpoints/M700_gino.pt`](checkpoints/M700_gino.pt). Round-2 ablations: `M700_gino_afno_lambda.pt`, `M700_gino_wno_lambda.pt`, `M700_gino_fno1d_lambda.pt`, `M700_gat_fno_lambda.pt`. Skip full-mesh Transolver / DeltaPhi / GIFNO-XT retrain.

---

## 13. n-scale M2100 (staged n3000 H5)

§11.2 gated M2100 after GINO-wide M1400 dropped IID **0.376 → 0.343**. The screen pack had only 133 H5; Box is now mounted. Incremental copy of missing names from `data/gifno_screen/n3000_h5_names.txt` into `data/gifno_screen/h5/` (2867 files; one local `run_370.h5` was a zeroed HDF5 and was recopied). Haskell `cache/n3000_seed42` is complete (`r_nom_signed.npy` + `fields.npy`, 1.1 GB). Nested-safe mix: n1000 train 700 + **1400 extras from n3000 outside the n1000 corpus** + OOD trains (3444 rows). Held-out tests stay seed-42 (150 / 144 / 144). Do **not** `make_splits(3000)`.

Lambda still has 133 H5; training reads the signed cache, not extra H5. Logs: new wandb project [`deeponet-nscale`](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-nscale) (`host=lambda`, tags `mix` / `encoder` / `fno_kind` / `host`, test metrics in `run.summary` only). Leave [`deeponet-residual`](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual) as the §10–12 archive.

Same wide + ReduceLROnPlateau recipe as §11.4 / §12: `--encoder {gno,gat} --fno --batch-size 32 --fno-width 64 --fno-modes 8,32 --fno-layers 4`.

| Mix | host | IID rel L2 / Pearson | dipping | 3-layer | ΔIID vs A10 M1400-wide 0.343 | wandb |
|-----|------|----------------------|---------|---------|------------------------------|-------|
| M1400-wide (control, §11.2) | A10 | **0.343 / 0.929** | 0.328 / 0.900 | 0.542 / 0.855 | — | [h94bwcl7](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-residual/runs/h94bwcl7) |
| M2100 GINO-wide + sched | A10 | 0.334 / 0.934 | **0.327 / 0.903** | 0.550 / 0.854 | **−0.009** | [sqqtczdc](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-nscale/runs/sqqtczdc) |
| **M2100 GAT+FNO + sched** | A10 | **0.326 / 0.933** | **0.325 / 0.901** | 0.541 / 0.861 | **−0.017** | [38pseh2m](https://wandb.ai/kurtwal98-university-of-california-berkeley/deeponet-nscale/runs/38pseh2m) |

GINO best-val epoch 28 / stop 108; GAT 44 / 124. Dipping stays \(\le 0.35\). IID keeps falling, but neither run hits the pre-declared **≥ 0.02** drop vs M1400-wide 0.343 (GAT is 0.003 short). Three-layer is still worse than laptop GINO **0.533** (GAT 0.541 is the closer of the two). **Skip M7680** (no rest-of-IID H5, no `n7680_seed42` Haskell).

Winner rule unchanged: best three-layer rel L2 among IID \(\le 0.371\) and dipping \(\le 0.35\). All M2100 runs qualify on the gates; both lose three-layer to [`checkpoints/M700_gino.pt`](checkpoints/M700_gino.pt). Extra IID moved IID from 0.343 → **0.326** (GAT) toward GIFNO-XT **0.30**, but that does not unseat the ship.

**Ship:** keep [`checkpoints/M700_gino.pt`](checkpoints/M700_gino.pt). Pulled `M2100_gino_wide_lambda.pt` and `M2100_gat_fno_lambda.pt`. Skip mesh Transolver / DeltaPhi / GIFNO-XT retrain.


