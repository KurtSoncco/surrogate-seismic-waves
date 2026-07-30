# AGENTS.md — TH-FNO physics-anchored TF surrogate

> Context file for the coding agent. Read fully before writing code each session.
> This encodes decisions already made. Do not relitigate them; if evidence
> contradicts a decision, surface it and STOP — do not silently redesign.

---

## 0. One-paragraph mission

Predict the 2D OpenSees transfer function **directly** at query coordinates
`|TF|(x, log f)` from the soil field (Vs, ζ) + physics latents, so the same
operator transfers to **new geometries** (n-layer, dipping, with/without RF)
at low sample cost. The scientific claim is **sample-efficient cross-geometry
generalization**, NOT "beat OpenSees on the training corner." RV is one OOD
probe, not the goal.

**Active path (Session N+1):** calibrated trend (`TREND_FREQ_SCALE≈0.938`) +
**validation gate B** on `direct` vs `log_mult` residual (seeded, one variable).
Do not claim a residual win until B1 excludes 0 and B2 shows no edge regression.
Default `THFNO_PREDICT_MODE=direct` until the gate says otherwise.

Model is **deterministic given the field/ξ**. UQ is a separate future paper.
Do not add distributional losses or variance heads to this model.

---

## 0.1 Spatial domain (train vs OOD)

Full OpenSees meshes are typically **`(nz, 1500)`** @ `dx=dz=1 m`
(BC flanks + variability window).

**Train / IID strip (current):** the **central variability window of width 500**

```
full x:     [0 …… 499 | 500 …… 999 | 1000 …… 1499]
                 BC    |  model NX=500 |    BC
slice:               X_SLICE = [BC_WIDTH : BC_WIDTH+500]
model field:         (C, NZ_MAX≤128, NX=500)
query TF:            (x_strip, log f) at 21 recorders (±150 m, 15 m spacing)
```

Config: `NX = LX_VARIABILITY = 500`, `BC_WIDTH = 500`,
`X_SLICE_START/END = 500:1000`. Depth is padded/cropped to `NZ_MAX=128`.

**OOD lateral-extent probes (planned):** keep the **same physics**, change
the strip — e.g. evaluate on a wider/narrower central crop, or (later) full
`nx≠500` domains. The 500-wide strip is the train prior; other widths test
whether the `(x, log f)` operator generalizes in lateral extent, not just
geometry type.

---

## 1. Non-negotiable invariants (unit-testable; wire these as asserts/tests)

1. **Hard gate exactness (residual mode only).** When `dip == 0` AND `σ² == 0`
   (no random field), residual must be exactly 0 and prediction == `H_1D(trend)`.
   Free unit test for residual ablation. Direct mode has no gate. Test:
   `tests/test_gate.py`.
2. **If any baseline is used, it is `H_1D(trend)`, NOT realization-column geomean.**
   Column geomean launders the realization and is near-exact on pancake fields
   (aHV=50) for reasons unrelated to the model. D3 opponent only.
3. **Amplitude floor in any log-domain loss.** `ln(max(|TF|, EPS))`, `EPS` set
   from the data noise floor. Never `ln|TF|` raw (nulls → −∞ dominate loss).
4. **ξ is gauge-locked to a fixed KL basis.** Meaningful within a fixed
   covariance structure only. Never assume ξ transfers across correlation
   hyperparameters without re-projection. Flag if a run mixes bases.

---

## 2. Architecture (decided — direct TF)

```
field(Vs, ζ) + geom on central strip (nz, 500)
        ──► shallow FNO encoder ──► field embedding
                                         │
target (x_strip, log f) ──► Fourier features ──► DeepONet trunk ─┤
ξ / physics (CoV, rH, aHV, …) ─────────────────► head MLP ──────┘
                                                       │
                                              Softplus → |TF|_pred(x, f)
```

Query every supervised recorder and every frequency: **direct `|TF|` for each
`(log f, x)`**. Trunk is DeepONet so other `(x, f)` remain queryable without
regrid.

**Decisions (do not reverse without new evidence):**

- **Predict mode under gate B.** Default `direct`. Residual `log_mult` on
  calibrated trend is the A/B arm — promote only if §6.1 passes.
- FNO encoder on the **500-wide central strip**, **SHALLOW**.
- DeepONet trunk (arbitrary `(x, log f)` querying) with Tancik Fourier features
  on coordinates (SIREN ablation later).
- Physics latents `(CoV, rH, aHV, seed emb)` → head if full KL ξ unavailable in
  H5 — document the gap; do not invent fake KL.
- Strengthen FNO pointwise skip `W` for local scattering.

**Dropped / deferred:** mesh GNO; K/M edges; learned latent geometry code;
distributional / CRPS / variance head. See `DEFER_GNO.md`.

---

## 3. Loss (decided)

Pick ONE amplitude domain in run config:

```
L = SmoothL1( A(|TF|_pred) , A(|TF|_true) )
  + λ_peak · SmoothL1 on peaks (same A domain)
  + λ_spec · SmoothL1( ∂_logf |TF|_pred , ∂_logf |TF|_true )
```

Default: `A = ln(max(|TF|, EPS))`. Log base terms separately in W&B.
Supervise **per-recorder** on the 500-strip (esp. edges), not center-only.

---

## 4. DIAGNOSTICS GATE — run BEFORE residual training

- **D1** EV of `H_1D_trend` per corner.
- **D2** Is ΔH smoother / lower-rank than H_2D? If not → STOP, predict TF direct.
- **D3** Bake-off: trend Haskell vs realization geomean vs Pretell vs GIFNO.
- **D4** RV-inclusive GIFNO retrain (coverage control) on Lambda.

Write `results/diagnostics/GO_NO_GO.md`. Await go/no-go before Phase 3 arch.

**Recorded 2026-07-29:** D2 **FAIL** on RV pancake (`results/diagnostics/GO_NO_GO.md`).
Default run mode is therefore **direct `|TF|`** (`THFNO_PREDICT_MODE=direct`).
Do not train the gated residual stack unless D2 is re-run with contradictory evidence.

---

## 5. Success metric + state of evidence (Session N+1 — supersedes prior §5)

Train on geometry A **on the central-500 strip**, eval / few-shot on B with
`{0,10,100,1000}` samples. Report **per-recorder**, especially edges.
**Headline metrics = mean ± std over seeds.** Single-run = "(1 run, indicative)".

OOD checklist (non-exhaustive):

| Probe | What changes |
|-------|----------------|
| RV pancake | `rH/aHV` corner (existing) |
| n-layer / dipping | geometry (capability H5) |
| **short-rH / steep-dip** | only place trunk/lateral thesis is testable |
| **Strip width** | same field, `NX ≠ 500` crop — lateral-extent generalization |
| No-field flat | sanity (residual mode gate; direct mode still report) |

### 5.1 Established

- Pancake D2 FAIL was a **baseline artifact** (attr trend resonance mistuned ~6%;
  `f_H2D/f_attr ≈ 0.94`). Peak-stretch: EV 0.585 → 0.86.
- **Calibration KEEP:** GIFNO median `f_truth/f_H1D` → `TREND_FREQ_SCALE ≈ 0.938`,
  `H_eff = H / scale`. Lifts RV trend EV 0.585 → 0.802.
- Mean-profile rebuild **REJECTED** (EV 0.539 < 0.585). Do not revive without
  understanding the impedance-contrast mechanism.

### 5.2 Hypotheses (NOT established)

- "log_mult residual beats direct (0.401 vs 0.585)" — **confounded** first A/B
  (loss domain + residual type + cal scale changed together; 1 seed). NOT a winner.
- "center metric spatially flat" — true for DIRECT ckpt (21-rec Pearson
  0.653–0.657). **Not verified** for residual winner (A/B table edge/center
  contradiction → §B2).

### 5.3 Untested thesis

Cross-geometry generalization. GIFNO IID has little lateral structure → **cannot**
test the arbitrary-`x` trunk. Only short-rH / steep-dip can.

### 5.4 Reporting invariants

- Every A/B changes **one** variable; else label delta "confounded".
- Always report per-curve median / p10 / p90 alongside `rel_c`.
- Never let one corner (esp. pancake) state an architecture law.

---

## 6. Work order (Session N+1 — supersedes prior §6)

### 6.0 Robustness BEFORE re-running B (C1–C3) — APPLY FIRST

1. **C1 Clamp `g·Δ`:** `Δ_eff = C * tanh(g·Δ / C)`, `C = ln(3)` (≤ 3× correction).
2. **C2 Zero-init residual head** (branch final Linear → 0) so `exp(g·Δ)=1` at step 0.
3. **C3 Per-term loss normalization** (EMA of |term| before λ). Spec was ~95% of loss.

### 6.1 VALIDATION GATE B — nothing downstream until this passes

**B1** Seeded decT A/B, 3–5 seeds, TWO arms only:
`direct + cal-trend` vs `log_mult + cal-trend` (matched everything else).
Report mean±std `rel_c` + per-curve median/p10/p90.
- Gate: CI on `(direct − log_mult) rel_c` **excludes 0** → residual wins → proceed.
  Includes 0 → keep direct, **STOP claiming residual win**.

**B2** 21-recorder sweep on the **residual** winner ckpt (not direct).
- If residual edges < direct edges: log-mult is degrading edges — investigate
  before proceeding (paper-reframing).

### 6.2 After B passes

1. Freeze recipe.
2. **D** short-rH / steep-dip: D1/D2 then train winning recipe, per-recorder eval
   (edges emphasized). Cal-trend EV should DROP; ΔH is where residual earns keep.
3. OOD: RV pancake + strip-width with frozen recipe.

### 6.3 Do-not-regress

- Hard gate: flat + no-field ⇒ pred == cal-trend. Unit test.
- Baseline = calibrated `H_1D(trend)` (`scale≈0.938`), NOT realization geomean.
- Log-domain loss requires amplitude floor.
- No distributional loss / variance head.
- No GNO, no K/M edge features, no learned latent — physics latents first.

Launcher: `bash lambda_b1_seeded_ab.sh --limit 2000 --epochs 80`
B2: `python diagnostics/eval_pearson_all_recorders.py --ckpt … --predict-mode residual`

---

## 7. Paths

```
experiments/TH-FNO/                 # this experiment
  AGENTS.md                         # this file
  haskell_baseline.py               # trend + realization helpers (D3 / ablations)
  context_features.py               # dip, impedance grad, gate
  model.py                          # TH-FNO direct (residual ablation)
  diagnostics/                      # D1–D4 scripts
  lambda_train.sh                   # GIFNO corpus on Lambda
~/seiskit/seiskit/theory/layered_1d_tf.py
~/seiskit/comparison/Response_Variability/
experiments/GIFNO-FDO-XT/
```

## 8. Dataset shapes (verified 2026-07-29)

```
# Full mesh (OpenSees / GIFNO H5)
Vs / ζ:                   (nz, 1500) @ dx=dz=1 m
BC flanks:                columns [0:500) and [1000:1500)
central variability:      columns [500:1000)  → NX=500 train strip

# Response_Variability OpenSees-2D (OOD probe; still use central-500 crop)
Vs_field / Damping_zeta:  (nz≈H+bedrock, 1500) e.g. (94, 1500)
transfer_function/AF:     (1000,) center recorder only in H5
freq:                     (1000,) ~0.1–10 Hz
params:                   Vs1, H, CoV, Vs2, rH=10, aHV=50, seed, sobol_id

# GIFNO-XT training corpus (contract; verify on Lambda when Box mounted)
Vs_realization_2D:        (nz≤128, 1500) → strip (NZ_MAX, 500) after pad/crop
tf_per_sample.npy:        (N, 21, 1000)  — 21 laterals on the strip, 15 m spacing
freq.npy:                 (1000,) 0.1–10 Hz logspace
model input:              (C, 128, 500)  — C includes Vs, ζ, x, z (+ dip, imped.)
queries:                  |TF|(x_i, log f_j) for i∈{21 recorders}, j∈{1000 freqs}
Sobol 6D:                 Vs1, H, CoV, rH, aHV, Vs2

# Capability OOD example (three_layer) — still 1500-wide full mesh
Vs_realization_2D:        (35, 1500) case_0.h5  → crop central 500 for IID-strip eval
```

> Agent: on first run with GIFNO_DATA_ROOT, load one sample and assert; if mismatch,
> correct THIS block and report the diff.
> When testing strip-width OOD, do **not** silently reshape the FNO to a new NX
> without an explicit experiment flag — record the NX used in W&B and RESULTS.
