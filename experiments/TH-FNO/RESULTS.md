# Results — TH-FNO (AGENTS diagnostics-first)

## Diagnostics gate (2026-07-29)

See [`results/diagnostics/GO_NO_GO.md`](results/diagnostics/GO_NO_GO.md).

| ID | Verdict |
|----|---------|
| D1 | EV(trend) ≈ 0.58 on RV pancake |
| **D2** | **FAIL** — ΔH higher-rank / rougher than H₂D |
| D3 | Pretell best (~0.20); trend ~0.45; GIFNO ~0.65 |
| D4 | SKIPPED (no GIFNO mount) — run on Lambda |

**Update (Session N+1):** D2 FAIL was a **baseline artifact** (attr trend
resonance mistuned ~6%). Calibrated `TREND_FREQ_SCALE≈0.938` lifts RV trend EV
0.585 → 0.802. Mean-profile rebuild rejected (0.539). Residual training is back
under **validation gate B** (seeded A/B), not claimed as a winner.

## First A/B (1 run each — CONFOUNDED, indicative only)

Linear loss + cal-trend, n=2000, e≤80, seed=42 (Lambda 2026-07-29):

| Arm | rel_c | pearson_c | rel_edge |
|-----|-------|-----------|----------|
| direct | 0.585 | 0.657 | 0.599 |
| log_mult residual | 0.401 | 0.861 | 0.377 |

**Not a result:** confounded / single-seed. Spec term was ~95% of unnormalized
loss (`spec≈82` vs `sL1≈0.35`). Resolve residual spatial flatness with B2.

## Robustness C1–C3 (applied before B re-run)

- C1: `Δ_eff = ln(3)·tanh(g·Δ/ln(3))` — residual ≤ 3×
- C2: zero-init residual branch final Linear
- C3: EMA per-term loss norm before λ

## Validation gate B (next)

```bash
# B1 — 5 seeds, matched cal-trend + linear + C1–C3; only residual mode differs
bash experiments/TH-FNO/lambda_b1_seeded_ab.sh --limit 2000 --epochs 80

# B2 — 21-recorder on residual winner ckpt
uv run --project experiments/GIFNO-FDO-XT python \
  experiments/TH-FNO/diagnostics/eval_pearson_all_recorders.py \
  --ckpt experiments/TH-FNO/checkpoints/th_fno_residual_s0/best_model.pt \
  --predict-mode residual --limit 2000
```

## Prior RV floor notes (still valid as OOD probe)

| Method | rel L2 mean | Pearson |
|--------|-------------|---------|
| Local-column Haskell | 0.360 | 0.872 |
| Trend Haskell | ~0.45–0.51 | ~0.75 |
| Pretell | 0.202 | 0.962 |
| GIFNO `xt_lat128_d128` | 0.599 | 0.591 |

## Mesh GNO / K–M

Deferred — see [DEFER_GNO.md](DEFER_GNO.md).
