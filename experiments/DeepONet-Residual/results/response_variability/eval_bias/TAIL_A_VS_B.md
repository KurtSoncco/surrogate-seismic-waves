# Pearson tail: coverage (A) vs seed variance (B)

The nested-test Pearson\(<0.9\) cases look like one CoV/\(r_H\) cloud. They are two mechanisms. The ceiling reframe is the **paper result**; the raw 20–33% tail count is not. Architecture fine-tunes and the importance-sampled corner OpenSees campaign are in §4 (SOTA ranking unchanged until a new ckpt wins nested Pearson/Anderson).

**Checkpoint.** `M7680_gino_rebal_ft.pt`  
**Tests.** Nested IID \(n=150\), dipping \(n=144\). Three-layer is out of scope here.  
**Reproduce.** `uv run python experiments/DeepONet-Residual/response_variability/diagnostics/tail_a_vs_b.py`

Tables: `corner_occupancy.csv`, `tail_6d_clusters.csv`, `seed_ceiling.csv`, `recorder_sensitivity.csv`, `part0_probes.csv`, `corner_is_locations.csv`.  
Figures: `sample50_iid.png`, `recorder_sensitivity_dipping.png`.

Primary Pearson remains **central recorder vs OpenSees**. Array-mean is a parallel check (it does not rescue the dipping tail).

---

## Headline: GINO beats the θ-only ceiling

Every 6D cell with ≥2 nested-test files: unordered OpenSees–OpenSees Pearson of \(|\mathrm{TF}|(f)\) on the central recorder, versus GINO–OpenSees on those same files. OPS–OPS is what a predictor that sees **only** θ can hope to match. GINO sees the realized field, so beating OPS–OPS is the claim that it extracts realization-specific structure.

| Domain | Replicated cells (files) | Cells with GINO \(>\) OPS–OPS | Median OPS–OPS | Median GINO–OPS | Median gap |
|--------|-------------------------:|------------------------------:|---------------:|----------------:|-----------:|
| IID | 28 (62) | **28 / 28** | 0.833 | 0.937 | **+0.10** |
| Dipping | 30 (143) | **29 / 30** | 0.850 | 0.917 | **+0.10** |

OpenSees does not agree with itself above ~0.83–0.85 at fixed θ. In the seed-sensitive corner it is much lower (0.54–0.65):

| Cell (pack indices) | CoV, \(r_H\) | OPS–OPS | GINO median / min |
|---------------------|-------------:|--------:|------------------:|
| IID 0, 72, 129 | 0.29, 62 m | **0.608** | 0.781 / 0.593 |
| IID 50, 148 | 0.28, 84 m | **0.652** | 0.748 / 0.541 |
| IID 6, 76 | 0.28, 77 m | 0.726 | 0.733 / 0.623 |
| IID 29, 112 | 0.26, 88 m | 0.694 | 0.838 / 0.821 |
| Dipping 10, 69, 83, 134 | 0.30, 78 m | **0.579** | 0.812 / 0.732 |
| Dipping 27, 58, 75 | 0.18, 91 m | **0.542** | 0.814 / 0.748 |

**Paper framing.** Not “GINO has a 20–33% failure rate below an arbitrary 0.9.” Rather: GINO exceeds OpenSees self-agreement in the overwhelming majority of the hardest, most seed-sensitive parameter cells (28/28 IID, 29/30 dipping, median +0.10 Pearson). The residual tail is the small minority of *realizations* where it does not. Report **fraction of replicated cells where GINO exceeds OPS–OPS** alongside Pearson/Anderson in the SOTA discussion (appendix is enough).

That is a generalization claim benchmarked against what is achievable at fixed θ, not against 0.9. A θ-only model should not be expected to beat ~0.83. A field-conditioned model should, and GINO does except on a handful of realizations (sample 50 is the documented case below the ceiling; §3).

Scalar \((\mathrm{CoV}, r_H, a_{HV})\) do not determine central \(|\mathrm{TF}|(f)\). More copies of the same θ will not shrink seed spread unless the encoder is told how the field actually turned out.

Shipped branch is **`xi_cov`**: realized recorder columns, ξ (KL of the GRF PSD; mode ranking uses \(r_H,a_{HV}\)), and scalar CoV. Scalar \(r_H\)/\(a_{HV}\) are not explicit channels. Encoder: 3-layer kNN=2 GNO on 21 recorders (~25 m spacing) → ~75 m receptive field vs population \(r_H\) up to 100 m.

---

## 1. The joint corner is combinatorially empty by construction

A Sobol design covers each axis well and the **joint** extreme cell poorly. That is arithmetic, not intuition, and it answers “why not more Sobol points?” with a number.

GIFNO IID design: **256** distinct 6D IDs. Independently varying axes, a “top 20% on every axis” cell occupies \(0.2^d\) of the design.

| Cell | Predicted unique IDs | Observed (n3000 screen) |
|------|---------------------:|------------------------:|
| CoV and \(r_H\) both in top 20% (\(0.2^2 \times 256 \approx 10\)) | ~10 | **13 / 256** |
| \((V_{s1}, H, \mathrm{CoV}, r_H, a_{HV})\) all in top 20% (\(0.2^5 \times 256 \approx 0.08\)) | ~0 | **0 / 256** |

Cuts below are **IID n1000 train** quantiles (CoV \(q_{50/75/80}=0.197/0.247/0.255\), \(r_H=53.5/75.3/79.9\) m), then applied to every split so the corner is not defined on the tail.

| Split | Files / unique 6D | CoV and \(r_H\) both \(>q_{50}\) | both \(>q_{75}\) | both \(>q_{80}\) | 5-axis all \(>q_{80}\) |
|-------|------------------:|----------------------------------:|-----------------:|-----------------:|----------------------:|
| IID train | 700 / 242 | 176 / 65 (25%) | 44 / 19 (6%) | 22 / 11 (3%) | **0 / 0** |
| IID test | 150 / 116 | 45 / 30 (30%) | 17 / 10 (11%) | 12 / 7 (8%) | 0 |
| IID Pearson\(<0.9\) | 30 / **26** | 17 / 13 (**57%**) | 8 / 6 (27%) | 5 / 4 (17%) | 0 |
| n3000 screen | 3000 / 256 | 802 / 69 | 237 / 21 | 154 / 13 (5% of IDs) | 0 |
| Dipping train | 672 / 32 | 194 / 9 | 61 / 3 | 42 / 2 | 0 |
| Dipping test | 144 / 31 | 34 / 9 | 14 / 3 | 9 / 2 | 0 |
| Dipping Pearson\(<0.9\) | 48 / **21** | 24 / 8 (**50%**) | 11 / 3 (23%) | 6 / 2 (12%) | 0 |

The IID “30 failures” are **26 unique 6D IDs**, three of them replicated in the tail (0/72/129, 6/76, 29/112). Dipping has 32 geometries in the whole campaign; the tail is 21 of them. The tail is *enriched* for the median CoV × \(r_H\) cell (57% / 50% vs 30% / 24% of the full test) but not confined there.

n1000 train already covers 242/256 Sobol IDs. Extra mix files are mostly RF replicates of IDs that already exist. M7680 extras cannot be 6D-counted locally (`n7680` meta missing). More *flat* Sobol reproduces the same sparse corner proportionally; it does not fill it.

---

## 2. Sample 50: the one documented miss below the ceiling

IID pack index 50 = `run_7514.h5` (`local_idx` 974, `rf_seed` 2486904). Same 6D as sample **148**. This is the case that sits *below* OPS–OPS (0.541 vs 0.652) while its twin sits well above (0.954). It is not a CoV/\(r_H\) quartile point.

| | Pearson (center) | Array-mean | leftover \(b\) | \(\Delta\ln A\) | Pearson high |
|--|-----------------:|-----------:|---------------:|----------------:|-------------:|
| 50 | **0.541** | 0.767 | **−0.008** | **−2.00** | 0.70 |
| 148 | **0.954** | — | **0.91** | −0.17 | 0.61 |

[`sample50_iid.png`](sample50_iid.png): coherent slow blob under the **center** of the strip. OpenSees \(|\mathrm{TF}|\) is a **needle resonance** near 1.5 Hz (\(|\mathrm{TF}|\gtrsim 10^2\)) plus a deep notch. GINO gets the frequency and misses two decades of amplitude. Leftover \(R\) is a ~200 spike; \(\hat R\) is ~30. Edge recorders on sample 50 are 0.87 and 0.92 — the miss is the center station, not the whole array.

**Architecture hypothesis (testable, not “rougher fields”).** A narrow spectral peak is exactly what a mode-truncated FNO (shipped 8 spatial × 16 frequency modes) smears. That is a sharper claim than generic high-band roughness, and it is why 50 is a case-study paragraph with the figure: the one documented instance where the model underperforms the realization-blind ceiling. Do not fold it into the CoV quartile forest. More Sobol of this same 6D will not fix a truncated mixer.

Cells 6/76 and 29/112 are the opposite: GINO is already *at or above* ceiling there. Those are the right probes for whether a receptive-field / realized-ACF change does anything *before* new OpenSees (§4).

---

## 3. Dipping array-mean was checked; it does not rescue the tail

Dipping breaks left–right symmetry, so station 10 is not obviously typical. We scored array-mean and edge recorders on the same packs.

| | Central \(<0.9\) | Array-mean \(<0.9\) | Tail median central / array | Rescued by array | Extra failures on array |
|--|-----------------:|--------------------:|----------------------------|-----------------:|------------------------:|
| IID \(n=150\) | 30 (20%) | 31 (21%) | 0.85 / 0.86 | 5 | 6 |
| Dipping \(n=144\) | 48 (33%) | **56 (39%)** | 0.86 / 0.85 | 8 | **16** |

The alternative explanation (central-only scoring as a symmetry-breaking artifact) **does not hold**. The dipping tail is slightly *larger* on array-mean, not smaller; 40 of the 48 central-tail cases stay below 0.9. IID control is unchanged (20% vs 21%). Report both in the paper. Keep central as the primary ranking metric; the array check is the good-faith control.

---

## 4. Roadmap (sequenced)

Architecture (receptive field + realized autocorrelation) and data (new corner locations) are both needed. 0a/0b are scored; 0c is training because sample 50 is still below the OPS–OPS ceiling. OpenSees H5s are on Box `data/corner_is/` (not mixed into the main `h5/` tree).

**Do not** change the primary SOTA ranking in §Headline until a new leftover ckpt actually wins nested Pearson/Anderson. Both 0a and 0b trip the three-layer kill (rel L2 0.556 / 0.541 > 0.533).

### 0 — Architecture on existing M7680 (Savio GPU)

Shipped recipe: init `M7680_gino_rebal_ft.pt`, mix M7680, `--iid-frac 0.34`, three-layer val stop. Two **named** fine-tunes so probes 6/76 and 29/112 are not confounded. M7680 is **legacy20**; FT init remaps ξ+CoV and pads ACF.

| Arm | Checkpoint | Change | Status |
|-----|------------|--------|--------|
| ship | `M7680_gino_rebal_ft.pt` | ξ seed-replay + CoV; kNN=2 GNO | scored (presentation pack) |
| 0a | `M7680_xi_field_acf_ft.pt` | `xi_field_acf`: field-FFT ξ + CoV + empirical ACF length; freeze GNO | done: job **38670232**, early stop ep 94 (best 3L rel L2 0.537 @ 14). Test IID / dip / 3L rel L2 **0.390 / 0.352 / 0.556** |
| 0b | `M7680_gno_rh_dilate_ft.pt` | r_H-dilated GNO skip \(d=\mathrm{clip}(\mathrm{round}(r_H/25),1,8)\); unfreeze GNO, encoder LR \(10^{-4}\) | done: job **38670233**, early stop ep 126 (best 3L rel L2 0.507 @ 46). Test IID / dip / 3L rel L2 **0.334 / 0.337 / 0.541** |
| 0c | `M7680_fno_modes832_ft.pt` | FNO modes 8×32, freeze GNO, new FNO head | **submitted** job **38722679** (`savio3_gpu` GTX2080TI): sample 50 still below 0.652 after 0a/0b |

Score:

```bash
uv run python experiments/DeepONet-Residual/response_variability/diagnostics/score_corner_probes.py
```

**Probe table** (IID pack indices; central Pearson vs OpenSees). 0a/0b raise the weak members of 6/76 and 29/112 except 0a on **112** (0.821 → 0.796). Sample 50 is closer to the ceiling but still below, so 0c is on.

| Ckpt | 6 | 76 | 29 | 112 | 50 | 148 | 50 vs OPS–OPS 0.652 |
|------|--:|--:|--:|----:|---:|----:|--------------------:|
| ship | 0.623 | 0.844 | 0.856 | 0.821 | **0.541** | 0.954 | below |
| 0a `xi_field_acf` | 0.724 | 0.903 | 0.952 | 0.796 | **0.629** | 0.807 | below |
| 0b rH-dilate | 0.815 | 0.884 | 0.906 | 0.861 | **0.615** | 0.961 | below |
| 0c FNO 8×32 | — | — | — | — | — | — | pending |

OPS–OPS on the same cells (ship pack): 6/76 = 0.726, 29/112 = 0.694, 50/148 = 0.652. Live numbers: `part0_probes.csv`.

### 1–2 — Importance-sampled corner OpenSees (parallel, GIFNO 6D)

Writer: `response_variability/corner_is_design.py`. Truncated box CoV \(\in[0.255,0.3]\), \(r_H\in[80,100]\); other axes usual GIFNO bounds. Large LHS, subsample 32 with \(p\propto\hat A(\theta)\) (kernel smoother of nested-test Anderson on \((\mathrm{CoV},r_H)\)). Excludes the **13** existing n3000 Sobol IDs in that cell.

| File | Rows | Notes |
|------|-----:|-------|
| `corner_is_manifest.csv` | **160** | 32 new 6D × 5 RF seeds |
| `corner_is_locations.csv` | 32 | `held_out=1` on **8** locations (never train): sample_id 2, 3, 6, 20, 21, 24, 28, 31 |
| `sample50_extra_seeds.csv` | **10** | exact sample-50 6D; seeds ≠ 2486904, 2823621 |
| `corner_is_all.csv` | **170** | concatenated array 0–169 |

Submit (Savio CPU, OpenSees env `~/seiskit`; else Stampede SKX):

```bash
sbatch experiments/DeepONet-Residual/hpc/savio_corner_array.sh
# fallback: sbatch .../stampede3_corner_array.sh
```

Savio array job **38664146** finished **170/170** with no failures. QC: every file matches the manifest (`sample_id`, `rf_seed`, CoV, \(r_H\)), `Vs_realization_2D` is finite on a 1500-column strip, and `recorders/accel/data` is \((n_t, 42)\). Copied to Box `…/Neural Operator/data/corner_is/` (H5s + CSVs), **not** into the main GIFNO `h5/` tree (would collide with `run_0.h5`…`). Next: signed-cache the 24 train locations (5 seeds) as an extra mix slice; score the 8 held-out locations as a true corner generalization set. Sample 50 is still below OPS–OPS after 0a/0b, so extra seeds at that 6D remain relevant once TFs exist.
