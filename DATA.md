# Training, validation, and test data

Datasets are **not committed** (`.gitignore` covers `data/*`, `*.h5`, `*.npy`, `**/cache/`). This file records what exists on disk, how splits are formed, and which experiment uses which protocol.

Inventoried **7 Sep 2026** from the Box mount at `/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data` and the local screen pack `data/gifno_screen/`.

## Where the files live

| Location | Role | On this machine |
|----------|------|-----------------|
| Box `…/Neural Operator/data/h5/` | Full IID OpenSees runs `run_0.h5`–`run_7679.h5` (7680 files) | Yes (FUSE) |
| Box `…/data/transfer_function/` | IID TF cache + POD bases + GIFNO checkpoints | Yes |
| Box `…/data/ood_dipping/` | 960 dipping-interface H5 + manifest | Yes |
| Box `…/data/ood_three_layer/` | 960 two-soil + bedrock H5 + manifest | Yes |
| `data/gifno_screen/` | Laptop pack: **3000** stratified IID H5 + **full 7680** TF cache + both OOD trees | Yes |
| `experiments/DeepONet-Residual/cache/` | Signed residuals, fields, seed-42 splits | Yes (~3.7 GB) |
| `data/1D Profiles/TF_HLC/` | FLAC 1D Vs / TTF pickles (~1000 samples) | **Missing** |

Point 2D experiments at a root that contains `h5/` and `transfer_function/`:

```bash
export GIFNO_DATA_ROOT="/path/to/data"          # Box or data/gifno_screen
export GIFNO_OOD_DIPPING="$GIFNO_DATA_ROOT/ood_dipping"
export GIFNO_OOD_THREE_LAYER="$GIFNO_DATA_ROOT/ood_three_layer"
```

DeepONet defaults to the Box path above if `GIFNO_DATA_ROOT` is unset. GIFNO also accepts `/mnt/box_lab/Projects/Neural Operator/data`.

---

## 2D OpenSees — in-distribution (IID)

Physics: 500 m soil-variability strip between 500 m absorbing pads (full domain 1500 m × variable depth). Random-field Vs on a **1-soil + bedrock** stack. Lateral recorders every 15 m about the strip center (21 stations). Transfer functions are Konno–Ohmachi smoothed FAS ratios, 1000 log-spaced bins from **0.1–10 Hz**.

### H5 contents (one run)

| Dataset / attr | Typical shape / range |
|----------------|------------------------|
| `Vs_realization_2D` | `(nz, 1500)`, `nz` = 25–110 |
| `Damping_zeta` | same as Vs |
| `Vs_profile_1D` | `(nz,)` (IID only) |
| `recorders/accel/data` | `(n_time, 42)` = 21 base + 21 surface |
| `params`: `Vs1`, `Vs2`, `H`, `CoV`, `rH`, `aHV`, `rf_seed` | CoV 0.10–0.30; H 15–100 m |

Models crop columns `[500:1000]` (variability strip) and pad depth to `NZ_MAX=128`. GIFNO / LOGLO input is `(4, 128, 500)` = normalized Vs, ζ, x, z. Residual DeepONet fields are `(3, 128, 21)` at recorder columns only (15 m spacing: indices `100, 115, …, 400`). Kernel-query GNO can still use denser **support** columns from the same 1 m Vs strip (`fields_support.npy`, stride 5 → 100 columns); labels stay the 21 OpenSees TFs. Do not interpolate those 21 TFs onto a denser x-grid and call it ground truth.

### TF cache

`transfer_function/tf_per_sample.npy` is **(7680, 21, 1000)** `float32` (~616 MB), aligned with `manifest.csv` (`sample_idx` 0…7679 = `run_{i}.h5`). Frequency axis `freq.npy`; recorder x-indices on the cropped 500-wide grid: 100, 115, …, 400.

IID parameter coverage (full 7680 for CoV/H/f0; n3000 cache for Vs/rH/aHV):

| Quantity | Min | Median | Max | Notes |
|----------|-----|--------|-----|-------|
| CoV | 0.100 | 0.199 | 0.300 | Sobol-flat |
| H soil (m) | 15 | 58 | 100 | Nearly uniform; thinner tail above 95 m |
| Bedrock (m) | 10 | 10 | 10 | Fixed |
| Vs1 (m/s) | 101 | 190 | 356 | Surface soil |
| Vs2 (m/s) | 767 | 1060 | 1464 | Bedrock |
| rH | 10 | 55 | 100 | RF correlation length |
| aHV | 10 | 22 | 48 | RF anisotropy |
| f0_effective (Hz) | 0.31 | 0.85 | 4.51 | Most mass below 1.5 Hz |

`xi_damp` in the residual cache is a constant 0.05.

**f0_effective audit (7 Sep 2026):** stored values are the *nominal 1-D* site frequency, exact on the trend profile.

| Corpus | Formula stored in `f0_effective` | Check | Caveat |
|--------|----------------------------------|-------|--------|
| IID | `Vs1 / (4 H_discretized)` | 0 relative error on n=3000; equals travel time through `Vs_profile_1D` | 2D TF peak at the center recorder is typically ~6% lower (heterogeneous RF) |
| Dipping | same, using **center** H | 0 error vs `H_discretized` | Ignores the dip: H varies by up to 26 m across the 500 m strip. Local column f0 is not stored. |
| Three-layer | `1 / (4T)`, `T = H1/Vs1 + H2/Vs_mid` | 0 error vs discretized H1, H2 | **Not** `Vs1/(4(H1+H2))`. Manifest `damping_freq_first` is capped at 3 Hz when f0>3 (720/960 runs). |

Use `f0_effective` for the nominal quarter-wave / travel-time frequency. Do not treat it as the OpenSees TF peak. Residual code `f0_quarter_wavelength` uses the same 1/(4T) definition (soil layers only, bedrock excluded).

### Local screen pack vs full corpus

`data/gifno_screen/` copies the **entire** TF cache (7680 rows) but only **3000** H5 files listed in `n3000_h5_names.txt`. Those 3000 are the nested CoV×H stratified subset `n1000 ⊂ n2000 ⊂ n3000` (seed 42), **not** `run_0`…`run_2999`. Full-corpus training still needs Box (or HPC) `h5/` for the remaining 4680 runs. There is **no** `n7680_seed42` Haskell cache on this machine.

Stage a pack: `experiments/DeepONet-Residual/stage_screen_pack.sh`.

---

## 2D OpenSees — OOD campaigns

Each campaign is **32 Sobol geometries × 30 RF replicates = 960** runs. Splits are 70 / 15 / 15, seed 42 → **672 / 144 / 144**. Held-out tests are these 144-file slices (never resampled when mix size grows).

| Campaign | Geometry | In-family with IID? | Extra observed variables |
|----------|----------|---------------------|--------------------------|
| `ood_dipping` | 1-soil + bedrock with a live dip across the 500 m strip | CoV, Vs1, Vs2, rH, aHV yes. H only 26–58 m. Bedrock **21–33 m** (IID is always 10). 1-layer Haskell nom still valid. | `dip_angle_deg` ∈ [−2.94, 2.85] (32 unique); `dip_direction` (sign of angle: 53% right_to_left, 47% left_to_right); `dip_span` = 500 m |
| `ood_three_layer` | **2-soil + bedrock** (two RF seeds) | CoV/rH/aHV yes (`CoV1=CoV2`, same for rH/aHV). Vs2/bedrock yes. Vs1 truncated (98–230). Total soil **11–23 m**. f0 median **3.50 Hz** (IID 0.85). Nom misspecified. | `Vs_mid` 461–551 m/s; `H1`,`H2` 5–12 m each; `Vs_contrast` = ln(Vs_mid/Vs1) ∈ [0.82, 1.59]; `seed1`,`seed2` |

OOD H5 files omit `Vs_profile_1D`. Residual caches live at `experiments/DeepONet-Residual/cache/ood_*_signed/` (`tf2d`, `r_nom_signed`, `fields`, …). Per-run GT TFs also exist under `cache/ood_*_tf/`.

Shared-variable overlap in short: CoV / rH / aHV / bedrock Vs match the IID Sobol box. Geometry does not — dipping H is a mid-depth slice of IID, three-layer columns are thinner than almost all IID profiles, and three-layer f0 sits above ~88% of IID.

---

## Split protocols (do not mix them)

All 2D work uses **seed 42** and **70 / 15 / 15**, but the *population* being split differs.

### A — GIFNO / LOGLO-POD: prefix of the manifest

`experiments/GIFNO/data_loader.py` takes `manifest[:limit]` then `torch.randperm` (seed 42).

| `--limit` | Train | Val | Test | Typical use |
|-----------|------:|----:|-----:|-------------|
| 500 | 350 | 75 | 75 | Smoke / profiling |
| **2000** | **1400** | **300** | **300** | LOGLO-POD screen (`tier2_pod64`) |
| 7680 (no limit) | 5376 | 1152 | 1152 | Publication LOGLO (`tier2_pod64_full`) |

This **n=2000 is the first 2000 manifest rows** (`run_0`…`run_1999`). Only **537 / 2000** overlap the DeepONet stratified `n2000_seed42`.

### B — Residual DeepONet: nested CoV×H subsets, then 70 / 15 / 15

Indices: `experiments/DeepONet-Residual/residual_target.py` `stratified_sample_indices` (4×4 CoV×H quantile bins, round-robin, seed 42), nested so `n1000 ⊂ n2000 ⊂ n3000`. Splits: `cache/splits/iid_n1000_seed42.npz` and `ood_*_seed42.npz`. Extra IID for larger mixes is taken from n2000 / n3000 / n7680 samples **outside the n1000 corpus**. Do **not** call `make_splits(2000)` — that leaks ~107 of the 150 n1000 test files.

Canonical held-out tests (frozen):

| Slice | n | File |
|-------|--:|------|
| IID | 150 | `splits/iid_n1000_seed42.npz` `test` |
| dipping | 144 | `splits/ood_dipping_seed42.npz` `test` |
| three-layer | 144 | `splits/ood_three_layer_seed42.npz` `test` |

IID n1000 split: **700 / 150 / 150**. Mix validation (unless `IID*` tag): IID val 150 + OOD val 144 + 144 = **438**.

| Mix tag | IID train | OOD train | Total train | Val |
|---------|----------:|----------:|------------:|----:|
| M700 | 700 (n1000 train) | 672 + 672 | 2044 | 438 |
| M1400 | 700 + 700 extras from n2000 | 672 + 672 | 2744 | 438 |
| M2100 | 700 + 1400 extras from n3000 | 672 + 672 | 3444 | 438 |
| M7680 | 700 + 6680 extras from n7680 | 672 + 672 | 8724 | 438 |
| IID2000 | 700 + 1000 n2000 extras | 0 | 1700 | 150 |
| IID7680 | 700 + 6680 | 0 | 7380 | 150 |

M7680 extras need the full 7680 H5 tree and a `n7680_seed42` Haskell pass (not present locally). Code: `experiments/DeepONet-Residual/mix_ladder.py`, `domain_splits.py`.

---

## 1D FLAC (soil–bedrock profiles)

Expected at `data/1D Profiles/TF_HLC/`:

- `Vs_values_1000.pt` — variable-length Vs, padded to 29 layers × 5 m
- `TTF_data_1000.pt` — 1000-point transfer functions
- `TTF_freq_1000.csv`
- `Rho_values_1000.pt` (PINO)

Generation: Latin-hypercube soil Vs ∈ [100, 760] m/s, bedrock Vs ∈ [760, 1500] m/s, 1–29 soil layers (`scripts/data_generation/1D Profiles/Vs_soil_bedrock.py`). Several 1D trainers also drop profiles with f0 ≥ 2 Hz.

**These files are not on this machine.** `wave_surrogate` FNO / PINO use a 50 / 25 / 25 split of the ~1000-sample FLAC set.

---

## Tensor cheat sheet

| Pipeline | Input | Target |
|----------|-------|--------|
| LOGLO-POD / GIFNO | `(4, 128, 500)` Vs, ζ, x, z on the strip | `(21, 1000)` or scattered TF grid |
| Residual DeepONet | fields `(3, 128, 21)` + stochastic vector; trunk `(x/λ, f*, …)` | signed `R_nom = TF_2D − TF_1D` at 200 train freqs, eval at 1000 |
| 1D FNO (`wave_surrogate`) | Vs length 29 | TF length 1000 |

POD readout (LOGLO): `pod_mean.npy` `(21, 1000)`, `pod_modes.npy` `(21, 64, 1000)` in the screen pack (`(21, 32, 1000)` also on Box).
