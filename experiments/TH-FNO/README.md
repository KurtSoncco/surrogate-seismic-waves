# TH-FNO

Direct `|TF|(x, log f)` on the **central 500-column strip** of the full
`(nz, 1500)` mesh — per [`AGENTS.md`](AGENTS.md).

D2 FAIL (2026-07-29) → default `THFNO_PREDICT_MODE=direct` (not residual-on-trend).
See [`results/diagnostics/GO_NO_GO.md`](results/diagnostics/GO_NO_GO.md).

Train data = GIFNO-XT corpus (`GIFNO_DATA_ROOT`). RV / capability / **strip-width**
variants = OOD probes.

## Domain

```
full mesh (nz, 1500)  →  crop columns [500:1000]  →  model (C, ≤128, 500)
queries: 21 recorders × 1000 freqs  →  |TF|(x_i, log f_j)
```

`NX≠500` crops are a planned lateral-extent OOD test (same fields, different strip).

## Quick start

```bash
cd experiments/TH-FNO

uv run --project ../GIFNO-FDO-XT pytest tests/ -q
uv run --project ../GIFNO-FDO-XT python diagnostics/run_d1_d3.py --max-seeds 10

# Direct |TF| train (Lambda + GIFNO data)
bash lambda_train.sh --limit 2000 --epochs 100

uv run --project ../GIFNO-FDO-XT python cross_geometry.py
```

## Layout

| Path | Role |
|------|------|
| `AGENTS.md` | Decisions (direct TF, central-500 strip, OOD list) |
| `model.py` | Shallow FNO + Fourier DeepONet → Softplus `|TF|` |
| `losses_th.py` | SmoothL1 on `ln(max(|TF|,EPS))` + peak + spectral |
| `gifno_dataset.py` | GIFNO corpus, central-500 crop |
| `diagnostics/` | D1–D4 |
| `DEFER_GNO.md` | GNO escalation gate |
