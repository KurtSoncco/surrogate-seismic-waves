# Baseline-fix D1/D2 (RV pancake)

Critique: attr-based 2-layer `H_1D(Vs1,H,Vs2,ξ=0.05)` may misbuild the trend,
so EV~58% and high-rank linear ΔH can be artifacts. This re-runs the decision tree.

n = 160 (max_sobol=32, max_seeds=5)

## D1 — EV comparison

| baseline | EV | mean rel L2 |
|----------|----|-------------|
| attr {Vs1,H,Vs2,ξ=0.05} | 0.5850 | 0.4514 |
| **mean profile (fitted ζ)** | **0.5388** | **0.4696** |

## Peak alignment (first resonance)

- median Δf/f (attr − H₂D): 0.0607 (abs median 0.0607)
- median Δf/f (meanprof − H₂D): 0.0582 (abs median 0.0582)

## D2 — centered rank + log-ratio

| baseline | rank H₂D (ctr) | rank Δ_lin (ctr) | rank Δ_log (ctr) | lin easier? | log easier? |
|----------|----------------|------------------|------------------|-------------|-------------|
| attr | 8.89 | 16.01 | 12.47 | False | False |
| meanprof | 8.89 | 16.42 | 14.06 | False | False |

Raw (mean-included) ranks for reference:
- attr: H₂D=3.03, Δ_lin=16.27
- meanprof: H₂D=3.03, Δ_lin=16.72

## Decision

**Branch:** `B_peak_offset_remains` → refined after follow-up

| claim | result |
|-------|--------|
| Rebuild from depth-averaged Vs → EV 90%+ | **No** (EV 0.585 → 0.539) |
| Peak offsets drive the low EV | **Yes** (~6% systematic; stretch → EV 0.86) |
| Original attr-linear D2 retire delta? | **No** — confounded by resonance bias |
| Log-ratio Δ after peak-aligned trend | **D2 PASS** (rank 7.0 < H₂D 8.9) |

**Prefer:** calibrate trend resonance (or train log-mult residual); do **not** prefer direct `|TF|` from the original attr-linear D2 alone.

## Caveat (unchanged)

This is still the RV pancake OOD corner. Mount `GIFNO_DATA_ROOT` and run
D1–D2 on short-rH / steep-dip hold-outs — that is where delta-learning
is supposed to win.

---

## Follow-up (after mean-profile miss)

Mean-profile rebuild **failed** the primary bet (EV 0.585 → **0.539**). Peak offsets stay ~6% for both attr and meanprof.

| oracle / correction | EV | rank Δ_lin (ctr) | rank Δ_log (ctr) | lin easier | log easier |
|---------------------|----|------------------|------------------|------------|------------|
| attr (original) | 0.5850 | 16.01 | 12.47 | False | False |
| attr + peak-frequency stretch | 0.8567 | 16.58 | 7.01 | False | True |
| local center column Haskell | 0.7231 | 25.47 | 12.76 | False | False |
| geo recorders (forbidden train baseline) | 0.8618 | 19.09 | 9.40 | False | False |

Notes:
- ξ sweep on 2-layer attr peaks at ξ≈0.06 with EV≈0.60 — **damping alone cannot explain the 58%**.
- Median f_H2D / f_attr ≈ 0.943 (**~6% systematic**); f_H2D / f_local ≈ 0.986 (~1.4%).
- Peak-stretch lifts EV to **~0.86** and improves ranks, but Δ still higher-rank than H₂D on this corner.
- Local/geo use *realization* columns (not a shared trend). High EV there shows Haskell↔OpenSees can align; the gap for *trend* is resonance mis-tuning of the coarse attrs, not “pancake physics is 42% residual.”

### Revised verdict

- **Primary suspect (attrs vs mean profile): REJECTED** for this dataset.
- **Peak-frequency offset: CONFIRMED** — original D2 on linear ΔH is **partially confounded** by a ~6% resonance bias; it is not a clean test of intrinsic ΔH complexity.
- **Original D2 not fully void** (meanprof EV did not hit 90%+), but **do not retire delta-learning** from attr-baseline D2 alone.
- Even after peak stretch / local / geo, centered Δ rank stays above H₂D on pancake → residual still not “easier” here; that may be corner-specific.
- **Next:** fix or calibrate trend resonance (effective H/Vs), then D1–D2 on GIFNO short-rH / steep-dip.
