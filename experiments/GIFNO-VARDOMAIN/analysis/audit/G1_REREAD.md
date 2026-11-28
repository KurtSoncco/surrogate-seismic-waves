# G1 reread

The generated top README was checked against `audit_summary.md` after the narrative was regenerated from `write_narrative.py`. No sentence was deleted for a missing audit item. The headline is unchanged: IID (720 records, 24 groups) and dipping (90 records, 3 groups) are positive in 0.4–10 Hz, IID mid-band pooled g log-L1 is 0.051321, and this is not lateral accuracy.

| README claim | Audit item |
| --- | --- |
| P128n128, epoch 46, float32 checkpoint dtype, blind outside the score, `val_score` 0.04987, epoch 46 is not asserted as the minimum of a missing curve | 1.7 |
| IID and dipping mid-band and high-band pooled g log-L1, harm, and pooled skill, above the measured floor | 1.4, 1.8 |
| That result is not lateral accuracy; 1000 m and 1500 m windows are context invariance of the 500 m strip | 1.2 |
| Three-layer stays out: proposal floor 0.0136, harm about one third, realization floor above the 1D error | 1.4, 1.5, 1.8 |
| 0.1–0.4 Hz is absolute error. Treasure Island below 0.4 Hz is not an accuracy result; the 128-mode basis keeps a fraction 0.0138 of that residual | 1.3, 1.6, 1.8 |
| Measured floor 0.000228 and the 15 s leakage 0.563 / 0.042 / 0.122. On the scored Treasure Island bands the prior exceeds the model by less than that leakage | 1.3, 1.4, 1.8 |
| Practice comparison is elastic pooled g log-L1. Linear \|TF\| Pearson is not reprinted. Cached rigid-base arms are not the comparison | Rescope the metric |
| Census counts, round-0 unpadded sections are not clean domains, `full_svlab.h5` is not a second solve | 1.1 |
| Every IID and OOD training window is 500 m, so the low-frequency lateral question has no other width | 1.2 |
| Manifold sections are pad-cropped, troughs are absent, blind widths are 2–4 sections, and blind versus train separation is the splits-table AUC. Per-width rows are not a resolved width result | 1.1, 1.7, 1.8 |
| Error budget uses the 1D prior, the model error, the realization floor, and the POD floor. The FE floor is unmeasured, so finite ratios stay provisional | 1.4, 1.5, 1.6 |
| Stage chain and the clipped three-layer bar are omitted | Rescope the chained stage sentence |
| Homogeneous, CoV = 0, mirror, and monotone rows | Not an audit claim. Items 1.1–1.8 do not withdraw them. They are the frozen consistency table |
| Stored center-station curves for epochs other than the saved checkpoint | 1.7 records epoch 46. The neighborhood table is the frozen Phase 2 comparison; the audit does not call epoch 46 the minimum |

The 5th–95th λ/W percentiles in section 1.2 (2.44–5.36, Treasure Island 1.13, manifold 1.55) stay in `coverage.provenance.json`. They are not a cell of `coverage.csv`, so the generated page does not reprint them. The sentence that remains is the 1.2 conclusion: the training windows are 500 m.
