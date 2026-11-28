# Compatibility on a 500 m strip

Three claims are supported. The learned correction is empirically compatible with the column on windows that repeat a 500 m strip. On the IID and dipping tests, that correction reduces pooled absolute log error in 0.4–10 Hz. Broad lateral accuracy, and accuracy on a real section below 0.4 Hz, are not established. A padded campaign at 500, 1000, 1500, and 2000 m is future work. This revision does not run it.

This page is generated from the frozen ladder, baselines, error budget, consistency tests, epoch neighborhood, Case 7 column rescore, and the Phase 1 audit. Provenance sidecars record the checkpoint, the float32 forward, and the aggregation. A definition that differs from an earlier draft is named from `analysis/G0.md` and is not reprinted as a second headline.

The surface transfer function is written

$$|\mathrm{TF}| = |\mathrm{TF}_{\mathrm{column}}|\,e^{g}.$$

The column is the elastic half-space outcrop of the soil profile under the station, with the damping OpenSees applied, matched to Rayleigh damping at the two frequencies of that record. The network returns only g. Setting g to zero is an identity of this residual parameterization: the prediction is then the column. It is not a guarantee that the network emits zero.

The reported model is P128n128, epoch 46, float32. The ladder pools every labeled station and resamples Sobol groups (`bootstrap_unit=master`). Skill, where it is printed, is the skill of that pooled mean. Bands are 0.1–0.4 Hz, 0.4–2.5 Hz, 2.5–10 Hz, and 0.1–10 Hz.

The reported checkpoint forward is float32. Training config amp is on. evaluate.load_model binds bfloat16 autocast only on CUDA. The float32 comparison is one centered 500 m window on cpu, autocast stayed off, and the maximum absolute g log-L1 difference is 0.000000. A GPU bfloat16 sweep of that autocast was not run.

On the frozen float16 cache, the maximum absolute log gap versus surface_over_outcrop is 0.001931 on IID, 0.001147 on dipping, and 0.001235 on three-layer. cache/meta.json does not name the engine. The check is one center column and one edge column from each test group, not every station.

## Headline

Paper 2 is positive for the IID test (720 records, 24 Sobol groups) and the dipping test (90 records, 3 groups) in 0.4–10 Hz: pooled g log-L1 and the upper end of the master bootstrap lie below the center-column outcrop and below the 9-column Pretell geometric mean. IID mid-band pooled g log-L1 is 0.051321. This is not a lateral-accuracy result. The 0.1–0.4 Hz band is absolute error only. Three-layer is outside the claim (mid-band prior 0.003480 under the 0.0136 skill floor, harm on 32.22% of records, realization floor 0.008243 above the 1D error). Treasure Island below 0.4 Hz is not an accuracy result. Manifold widths and synthetic 1000/1500 m windows are compatibility. The training strip is 500 m.

The same pooled point estimates on those two tests and those two bands also lie below g = 0, the stored column prior. Intervals below are the master-bootstrap interval of the pooled mean.

| Test | Band | P128n128 (interval) | g = 0 | Center column | Pretell, 9 columns | Harm | Pooled skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IID | 0.4–2.5 Hz | 0.051321 (0.037233–0.068929) | 0.113464 | 0.139280 | 0.098023 | 0.97% | 0.5477 |
| IID | 2.5–10 Hz | 0.156935 (0.133382–0.183524) | 0.226893 | 0.333263 | 0.266441 | 0.00% | 0.3083 |
| Dipping | 0.4–2.5 Hz | 0.034362 (0.017507–0.046266) | 0.074034 | 0.105386 | 0.082995 | 0.00% | 0.5359 |
| Dipping | 2.5–10 Hz | 0.130513 (0.108037–0.152806) | 0.165686 | 0.250852 | 0.211429 | 0.00% | 0.2123 |

IID mid-band harm is 0.97% of 720 records. Dipping mid-band harm is 0.00% of 90 records. Harm is the share of records whose pooled g log-L1 is worse than g = 0.

Ladder provenance: checkpoint epoch 46, float32, aggregation `master`, skill reduction `pooled`. The Pretell mean uses the nine columns listed in `ladder.provenance.json`. Sample-id hash `bead3c98f13f`.

## What is outside that sentence

Three-layer stays out. Its mid-band prior sits under the proposal floor, so the ladder suppresses skill. About one third of those records are harmed, and the realization floor exceeds the 1D error. Absolute mid-band g log-L1 is 0.003098 (interval 0.002018–0.004711, 90 records, 3 groups).

The 0.1–0.4 Hz band is reported as absolute error. It is not part of the positive test.

| Test | Pooled g log-L1 | Interval | g = 0 |
| --- | --- | --- | --- |
| IID | 0.010738 | 0.004455–0.022489 | 0.017333 |
| Dipping | 0.003291 | 0.002195–0.004366 | 0.006594 |
| Three-layer | 0.001085 | 0.000686–0.001556 | 0.001112 |

The measured 1D-convention floor is 0.000228 on 2.5–10 Hz, a lower bound on max(FE floor, 1D-convention uncertainty) because the FE floor was not measured (`noise_floor.json`). Halving the 15 s Treasure Island record moves log|TF| by 0.563 (0.1–0.4 Hz), 0.042 (0.4–2.5 Hz), and 0.122 (2.5–10 Hz).

## Practice baselines

The in-family comparison is the table above: pooled g log-L1 against g = 0, the center-column elastic outcrop, and a 9-column geometric-mean outcrop. Both outcrops are elastic half-space calculations with Q(Vs) matched to Rayleigh damping at the record frequencies. Cached rigid-base arms are not the comparison.

Practice Pearson is of log|TF| only. G0 names the difference: the previous box used Pearson of linear |TF| on the full spectrum, then a median across records. That box is not reprinted, and its Pearson values are not placed beside this pooled log-L1.

The bars in `03_state_of_practice/central.pdf` are that elastic pooled comparison, read from `ladder.csv` and `baselines.csv`.

## Compatibility, and the coverage of the training strip

The training strip is 500 m. A window of 1000 m or 1500 m on the IID, dipping, and three-layer tests repeats the edge column of that strip. The change in pooled g log-L1 is context invariance. It is compatibility with the column, not a lateral-accuracy result. `04_width_and_manifold/central.pdf` plots that invariance from `consistency_tests.csv`. The previous width-skill curve is not in the main text.

Mid-band pooled g log-L1 when the same stations are placed in a wider window:

| Family | 500 m | 1000 m | 1500 m |
| --- | --- | --- | --- |
| Horizontal | 0.051321 | 0.051713 | 0.051863 |
| Dipping | 0.034362 | 0.034796 | 0.035052 |
| Three-layer | 0.003098 | 0.003051 | 0.003072 |

## Case 7, corrected column

The controlled width diagnostic is OpenSees versus the corrected elastic column on the existing Case 7 H5s. The network was not scored. There is no CUDA in this environment, and a new forward was not run. full_svlab.h5 was not read. Pre-fix Pearson tables are not reprinted.

Hallal, Cox, and Vantassel (JGGE 2023) and Hallal's dissertation are the domain rationale already used by this project: variability along at least 600 m for the fundamental-mode drop, about 1700 m for the other basin effects, and ground out to about 1000 m from the sensor. Those lengths were not computed in this repository.

Mean |log| gap on stations every 5 m. FFT window 1310.72 s. Input scale 1.9777, the acceleration value recorded with that window.

| Crop | 500 m, 0.1–0.4 Hz | 1500 m, 0.1–0.4 Hz | 500 m, 0.4–2.5 Hz | 1500 m, 0.4–2.5 Hz |
| --- | --- | --- | --- | --- |
| Flat west | 0.0243 | 0.0783 | 0.0215 | 0.0185 |
| Center | 0.1029 | 0.1007 | 0.0522 | 0.0594 |
| Thinning | 0.3743 | 0.1621 | 0.0796 | 0.0650 |
| Ramp | 0.0810 | 0.1458 | 0.0297 | 0.0486 |
| Flat east | 0.0267 | 0.1441 | 0.0193 | 0.0445 |

The low-band gap is not monotone in width. A 2000 m row is a window of full_qvs.h5. Nonzero mesh_shift_m means the window was slid onto the 4000 m mesh and meets the domain edge. Open markers on `04_width_and_manifold/case7_column.pdf` are those rows. They are not a padded 2000 m domain, and they are not a network score.


Census: IID 7680 records in 256 groups, split 6240/720/720, a 500 m strip of a 1500 m mesh, depth up to 110 m. Dipping 960 records, split 780/90/90. Three-layer 960 records, the same split. Manifold cache 301 sections, split 250/35/0/16.

Coverage is the per-record table `coverage.csv`, with the summary in `coverage.provenance.json`. Every IID and OOD training window is 500 m, so a low-frequency lateral question sits where that density has no other width.

The coverage figure is `04_width_and_manifold/coverage.png`, copied from the audit.

Manifold sections are pad-cropped. Troughs are absent from the cache that was scored. Blind counts per width are 2–4 sections. Those rows are compatibility.

| Split | Width | n records | Mid-band pooled g log-L1 | Interval |
| --- | --- | --- | --- | --- |
| val | pooled | 26 | 0.064264 | 0.044013–0.086067 |
| blind | pooled | 11 | 0.030346 | 0.016069–0.050571 |
| blind | 500 m | 4 | 0.056314 | 0.030136–0.082492 |
| blind | 1000 m | 2 | 0.024017 | 0.009512–0.038522 |
| blind | 1500 m | 2 | 0.049114 | 0.016408–0.081821 |
| blind | 2000 m | 3 | 0.014483 | 0.006119–0.029391 |

On this pooled reduction the blind mid-band g log-L1 is 0.030346 and the validation mid-band g log-L1 is 0.064264. Blind was not in the selection score. A logistic classifier separates manifold blind from manifold train at AUC 0.768 on 16 sections (`splits.csv`). The blind pool is small, and the per-width rows are not a resolved width result. `06_reliability_and_broadband/central.pdf` shows the same pooled means.

## Error budget

For each family and band the budget is the 1D-prior error, the error left after P128n128, the realization floor, and the POD floor. The headline ratio is (1D error − model error) / (1D error − max of the measured floors). The FE floor is unmeasured, so every finite ratio is provisional. A nonpositive gap is omitted.

On IID and dipping, in the mid band, the high band, and overall, the realization floor exceeds the 1D error, so those ratios are omitted. Finite ratios that remain are marked `provisional_fe_unmeasured`. The figure `06_reliability_and_broadband/error_budget.pdf` is drawn from `error_budget.csv` only. The FE floor is absent from the figure because it was not measured.

| Family | Band | 1D prior | P128n128 | Realization | POD | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| IID | low | 0.017333 | 0.010738 | 0.009699 | 0.009584 | 0.864 |
| IID | mid | 0.113464 | 0.051321 | 0.136414 | 0.004015 | omitted |
| IID | high | 0.226893 | 0.156935 | 0.258451 | 0.002958 | omitted |
| IID | overall | 0.118671 | 0.070896 | 0.134028 | 0.005373 | omitted |
| Dipping | low | 0.006594 | 0.003291 | 0.003705 | 0.003827 | 1.194 |
| Dipping | mid | 0.074034 | 0.034362 | 0.136511 | 0.002099 | omitted |
| Dipping | high | 0.165686 | 0.130513 | 0.265013 | 0.001544 | omitted |
| Dipping | overall | 0.081322 | 0.053951 | 0.131672 | 0.002452 | omitted |
| Three-layer | low | 0.001112 | 0.001085 | 0.000608 | 0.001471 | omitted |
| Three-layer | mid | 0.003480 | 0.003098 | 0.008243 | 0.001261 | omitted |
| Three-layer | high | 0.105369 | 0.066386 | 0.188157 | 0.001424 | omitted |
| Three-layer | overall | 0.033436 | 0.021542 | 0.061453 | 0.001373 | omitted |
| Manifold blind | low | 0.005321 | 0.001892 | — | 0.002385 | 1.168 |
| Manifold blind | mid | 0.047617 | 0.030346 | — | 0.002467 | 0.383 |
| Manifold blind | high | 0.114351 | 0.038119 | — | 0.000739 | 0.671 |
| Manifold blind | overall | 0.054973 | 0.024121 | — | 0.001922 | 0.582 |
| Treasure Island | low | 0.070577 | 0.070095 | — | 0.069601 | 0.494 |
| Treasure Island | mid | 0.046793 | 0.044108 | — | 0.004239 | 0.063 |
| Treasure Island | high | 0.112859 | 0.060592 | — | 0.003461 | 0.478 |
| Treasure Island | overall | 0.073837 | 0.056892 | — | 0.023679 | 0.338 |

Treasure Island rows in that table use the pooled mean. The interval stored beside them is a moving-block median and is not an interval for the mean (`error_budget.provenance.json`).

## Consistency

These tests are results. A failure is reported here, not moved to a footnote. Cutoffs are the proposals in `gates.py`: homogeneous median L1(|g_pred|) against 0.0136, mirror station-matched L1 against 0.1 times the median L1(g_true), monotone sign agreement 0.80.

| Test | Band | Value | Reference | Result |
| --- | --- | --- | --- | --- |
| homogeneous_column | low | 0.002648 | 0.013600 | pass |
| homogeneous_column | mid | 0.009882 | 0.013600 | pass |
| homogeneous_column | high | 0.034015 | 0.013600 | fail |
| homogeneous_column | overall | 0.015368 | 0.013600 | fail |
| cov0 | low | 0.002867 | 0.000000 | fail |
| cov0 | mid | 0.011676 | 0.000000 | fail |
| cov0 | high | 0.062455 | 0.000000 | fail |
| cov0 | overall | 0.024309 | 0.000000 | fail |
| mirror | low | 0.001507 | 0.000875 | fail |
| mirror | mid | 0.021830 | 0.009553 | fail |
| mirror | high | 0.048428 | 0.020572 | fail |
| mirror | overall | 0.024231 | 0.010477 | fail |
| monotone_pairs | — | — | 0.800000 | not run |

The homogeneous column passes 0.1–0.4 Hz and 0.4–2.5 Hz and fails 2.5–10 Hz: median L1(|g_pred|) is 0.034015 against the 0.0136 proposal. It also fails on 0.1–10 Hz. CoV = 0, built by repeating the center column of every IID test record, is worse than g = 0 on every band: g = 0 has log-L1 0 on that repeated column, and the network residual is the value in the table. The mirror test exceeds 0.1 times L1(g) on every band. Monotone pairs were not run: no existing pair changes only CoV or only dip, and a new finite-element pair was not solved.

g = 0 remains the identity of the residual form. The CoV = 0 rows are the measurement that the network does not enact that identity on a repeated column.

## Epoch 46

Epoch 46 is the stored `best_model.pt`. `val_score` is 0.04987. The selection that stored it averages IID and manifold validation log-L1 (`select_families` iid,manifold), query dtype float32. Blind was not in the score. `history.csv` is absent from the checkpoint, so this page does not say epoch 46 is the minimum of a curve.

Epochs 43, 44, 45, 47, 48, and 49 are not on disk and were not trained for this comparison. The four-checkpoint mean was not recomputed. Stored center-station curves exist for IID and dipping only. Those center-station pooled means are a different station set from the all-station ladder, and they are not a validation horse-race among the four checkpoints.

| Family | Curve | Band | Center-station pooled g log-L1 |
| --- | --- | --- | --- |
| iid | P128n128 | mid | 0.052029 |
| iid | P128n128 | high | 0.157770 |
| iid | ensemble_mean_g | mid | 0.046713 |
| iid | ensemble_mean_g | high | 0.150403 |
| ood_dipping | P128n128 | mid | 0.035194 |
| ood_dipping | P128n128 | high | 0.132982 |
| ood_dipping | ensemble_mean_g | mid | 0.033915 |
| ood_dipping | ensemble_mean_g | high | 0.126384 |

The all-station IID mid-band ladder value is 0.051321. The IID center-station row above is index 10 only. The two reductions are not averaged.

## Treasure Island

Treasure Island is one 4000 m section, `full_qvs.h5`, a 15 s Q(Vs) record. Each scored station is the center of a 2000 m window. The forward used for the ladder was not rerun in this revision; g is the stored `treasure_g.npz` named in the provenance sidecar. The FFT window on that scoring is 1310.72 s. The bootstrap unit is `moving block along x`. The interval in `ladder_treasure.csv` is the median of station g log-L1. The pooled mean is a separate column. Those two reductions are not combined into one interval.

Blind selection: train.selection_score for the unified stage is manifold validation log-L1 only, and blind is logged rather than selected. This checkpoint's config select_families is the p128 branch: the mean of IID and manifold validation log-L1, with bf16 agreement under AGREE_FLOOR 0.0136 and query dtype float32. Blind is not in select_families. history.csv is not in the checkpoint directory, so the claim that epoch 46 is the minimum of the curve was not recomputed. Treasure Island was not a selection input.

On the Phase 1 distance audit, Treasure Island's nearest-neighbor distance to the IID training cloud is 1500.008, percentile 1.00 among 900 synthetic test points (`splits.csv`). It sits at the far edge of those distances.

Pooled g log-L1 on the 200 stations:

| Band | P128n128 | g = 0 | Status |
| --- | --- | --- | --- |
| 0.1–0.4 Hz | 0.070095 | 0.070577 | not_an_accuracy_result |
| 0.4–2.5 Hz | 0.044108 | 0.046793 | scored_band |
| 2.5–10 Hz | 0.060592 | 0.112859 | scored_band |
| 0.1–10 Hz | 0.056892 | 0.073837 | not_an_accuracy_result |

Below 0.4 Hz the pooled model error is 0.070095 against a prior of 0.070577. The 128-mode basis keeps a fraction 0.0138 of that low-band residual (`representation.csv`). Halving the 15 s record moves the band by 0.563. The mismatch is strongly visible on the figure. The 15 s record does not validate that band, so it is not an accuracy result.

On the scored mid band the median of the 200 stationwise skills is 0.0927. One minus the ratio of the separate median log-L1 values, 0.0304 and 0.0321, is 0.0522. The skill of the mean errors is 0.0574. The median skill is not that ratio.

On the scored bands the prior in the table above exceeds the model. Both differences are smaller than the 15 s record-length leakage on those bands (0.042 and 0.122).

`05_treasure_island_full/central.pdf` marks 0.4 Hz and shows g_pred − g_true from the stored fields. Hatched bars in panel (d) are the low band and the overall band, which includes it. The center-column and Pretell outcrops for this section are in `baselines.csv`; they are a different column from the stored prior and are not drawn as an accuracy comparison.

## Dropped

Round-0 unpadded 2000 m sections include the absorbing boundary. `ladder.csv` records them as `not_scored_unpadded`. They are not clean 2000 m domains.

`full_svlab.h5` matches `full_qvs.h5` surface motion. The ladder records it as `not_a_second_solve`. It was not scored, and it is not a damping experiment.

G0 names a further difference and this page does not rebuild it: an earlier stage sentence chained a mean of per-station skills to the skill of a pooled error, and a three-layer stage bar was a mean of per-station skills clipped on the plot. Those reductions are omitted. The figure in `06_reliability_and_broadband/` is the pooled g log-L1 only.

## What a wide-domain claim still needs

Current machines cannot run padded 2000 m widths, and an unpaired smaller solve would not answer the question. The missing labels are paired master geologies at 500, 1000, 1500, and 2000 m, with 500 m repeated-edge pads, depth through 250 m, records of 200–400 s, gradient and layered Vs families, a CoV = 0 slice, and damping varied independently of Vs. A positive variable-domain claim needs those labels and a model trained from scratch in `experiments/GIFNO-VARDOMAIN/` with checkpoints in `checkpoints/vardomain/`. The present checkpoint is not that validation.

## Where the figures are

- `01_method_overview/` — the column is kept, and g = 0 is an identity of the residual form.
- `02_tf_heatmaps_and_skill/` — stored 500 m exemplars, dipping run 217 and blind section man_0297. The caption states the selection rule. The fields were not redrawn.
- `03_state_of_practice/` — elastic pooled g log-L1 against g = 0, the center column, and the 9-column Pretell mean. The mixed-prior practice box is not this figure.
- `04_width_and_manifold/` — context invariance, the Case 7 column diagnostic, and the audit coverage figure.
- `05_treasure_island_full/` — 0.4 Hz marked, g_pred − g_true shown. The low band is not an accuracy claim.
- `06_reliability_and_broadband/` — pooled absolute g log-L1, validation beside blind, the error budget, and the consistency table above.

The appendix states the epoch rule, the stored center curves, and the two files that were not scored. It does not add a second headline.

