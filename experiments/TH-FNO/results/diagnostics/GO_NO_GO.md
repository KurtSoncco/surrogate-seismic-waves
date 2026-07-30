# GO / NO-GO (diagnostics D1–D4)

Generated 2026-07-29. Evidence base: Response_Variability pancake corner
(`rH=10`, `aHV=50`) + no-field flat invariant. GIFNO Sobol stratified corners
pending `GIFNO_DATA_ROOT` mount.

## D1 — Explained variance of H_1D(trend)

| corner | n | EV_trend | rel_l2_trend_mean |
|--------|---|----------|-------------------|
| rv_pancake | 160 | 0.5850 | 0.4514 |
| nofield_flat | 1 | 1.0000 | 0.0000 |

Trend explains ~58% of H₂D variance on RV pancake; remaining residual is large.

## D2 — Is ΔH easier than H_2D?

| corner | rank_H2D | rank_delta | hf_H2D | hf_delta | delta_easier |
|--------|----------|------------|--------|----------|--------------|
| rv_pancake | 8.89 | 16.01 | 0.0002 | 0.0010 | **False** |
| nofield_flat | nan | nan | 0.0000 | 0.0000 | False |

**D2 gate (rv_pancake): `FAIL`**

ΔH has **higher** participation-ratio rank and **higher** high-frequency energy
than H₂D → residual learning is the harder problem on this corner.

## D3 — Bake-off on RV (mean rel L2 vs OpenSees-2D)

| Method | rel L2 |
|--------|--------|
| H_1D(trend) | 0.4514 |
| Realization local column | 0.3495 |
| Realization geomean (D3 opponent only) | 0.2431 |
| Pretell | **0.2006** |
| GIFNO grf_2d | 0.6533 |

**Number to beat on this corner:** Pretell ≈ 0.20 (geomean ≈ 0.24 is forbidden
as a training baseline per AGENTS §1.2).

Artifacts: `d1_d2_per_corner.csv`, `d3_bakeoff_rv.csv`, `d2_pass.json`.

## D4 — RV-inclusive GIFNO-XT coverage control

**STATUS: SKIPPED on this host** — `GIFNO_DATA_ROOT` / `tf_per_sample.npy` +
`manifest.csv` not mounted.

Lambda procedure (coverage vs architecture):

```bash
export GIFNO_DATA_ROOT=~/gifno_data
bash experiments/TH-FNO/lambda_d4_gifno_rv_inclusive.sh --limit 1000 --epochs 50
# Then score the D4 checkpoint on RV; if RV recovers → coverage; else architecture.
```

Until D4 completes, coverage vs architecture remains **undecided**. That does
**not** reopen D2: residual learning is already gated off by D2 FAIL.

## Decision (final)

**NO-GO for §2 residual / gated-delta training.**

| Gate | Result | Action |
|------|--------|--------|
| D2 | FAIL | **STOP residual stack**; predict `|TF|` **direct** |
| D4 | SKIPPED (no GIFNO mount) | Run on Lambda when data available; does not un-block residual |

Pivot implemented in code:

- `THFNO_PREDICT_MODE=direct` (default) — shallow FNO + Fourier DeepONet + physics
  latents → Softplus `|TF|` head; SmoothL1 on `ln(max(|TF|,EPS))` + peak + spectral.
- Residual mode retained only behind `THFNO_PREDICT_MODE=residual` for explicit
  ablations if D2 is revisited with contradictory evidence.
- Gate exactness invariants remain tested for residual mode.

Train path: `bash experiments/TH-FNO/lambda_train.sh` (requires GIFNO corpus).
---

---

## Addendum: baseline-fix decision tree

See [`GO_NO_GO_BASELINE_FIX.md`](GO_NO_GO_BASELINE_FIX.md).

- Mean-profile rebuild **did not** fix EV (0.585 → 0.539).
- ~6% systematic peak offset confirmed; peak-stretch EV → **0.86**.
- After peak-align: **log-ratio** ΔH centered rank **passes** D2 (7.0 < 8.9); linear Δ still fails.
- Do **not** retire delta-learning from the original attr-linear D2. Next: resonance-calibrated trend + GIFNO short-rH / steep-dip corners.
