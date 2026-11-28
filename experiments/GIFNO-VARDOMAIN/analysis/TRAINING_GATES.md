# Phase 6 — next model, not Paper 2

This note prepares a later training run. It does not start one, and it does not submit a job. A model trained under these rules is not part of Paper 2. Paper 2 stays on the frozen P128n128 results.

The frozen consistency tests already failed, and those failures belong to the checkpoint. On the homogeneous column the median L1 passes 0.1–2.5 Hz and fails 2.5–10 Hz (0.034 against the proposal 0.0136). CoV = 0 is worse than g = 0 on every band. The mirror station-matched L1 exceeds 0.1 times a typical L1(g) on every band. Phase 6.1 asks a different question: whether the augmentation itself is exact, not whether P128n128 obeys it.

The numbers below are the proposals in `gates.py`. They were not refit.

## 6.1 Augmentations

### Mirror

Training mirrors a window with probability one half. `WindowSet.__getitem__` draws that coin when `mirror` is on and the record's `allow_mirror` is true (`data.py`). The flip is `apply_mirror`: Vs and zeta columns reverse, and a station at x is queried at `width - x`. The transfer-function row stays with the station.

`tests/test_mirror_identity.py` checks that identity on the arrays, checks that `make_window` encodes those reversed arrays, checks that the coin matches the flipped window, and checks that `allow_mirror` false and `WindowSet.mirror` false leave the arrays unflipped. A second application of the flip restores the arrays.

The same test runs one mirrored column pair through `section_1d.surface_over_outcrop`. The outcrop of column j equals the outcrop of the mirrored column `n - 1 - j`, and the two columns are not the same soil. That is the linear SH operator on a column. It is not a 2D finite-element pair.

Finite-element confirmation is still outstanding. Five new OpenSees pairs were not solved, and none were required for this check.

### Length scaling

Length scaling was already checked on the 1D column in `experiments/case7a/tests/test_pilot_phase4.py` (`test_length_scaling_hysteretic_and_rayleigh_anchors`). It is not repeated here. When lengths scale by k and velocities do not, `TF_k(f) = TF(k · f)` for hysteretic damping, including Q(Vs). Rayleigh matches that identity only when `(f1, f2)` scale by k.

A 500 m run scaled by k = 4 is a 2000 m stack that is also four times deeper. It is a deeper geology. It is not the same soil widened.

### Velocity scaling

Velocity scaling does not exist in the training code, and it was not added. It would be valid only for explicit damping, never for Q(Vs).

## 6.2 Data

No new finite-element set was built. The Phase 5 width mix is not available, because that campaign was not run in this revision. Empty cells from the coverage audit and the worst cells from the error audit were not filled. Coarse-mesh pretraining stays off: the audit's finite-element floor is unmeasured, so there is no evidence that a coarse mesh sits inside the noise floor on 0.1–2.5 Hz.

Splits stay grouped by Sobol id or by manifold `group_id`. The Phase 1.7 audit already reproduced `split_by_group` and `manifold_holdout` with no group crossing. The blind set, manifold blind plus Treasure Island, is not a selection input. `selection_score` never reads a `manifold_blind` bin. `train_monitor.py` writes blind rows to `blind_sealed.csv` and will not open that file unless `open_blind` is true.

## 6.3 Loss and selection

The default loss is still `--loss log_l1`. The default protocol is still `unified`. With the new flag left off, which is the default, checkpoint selection is unchanged.

Unified selection is the unweighted mean of manifold validation `log_l1` (`selection_score`). IID, dipping, three-layer, and the blind manifold are logged and do not enter that score. Protocol `p128`, the branch that produced P128n128, is the unweighted mean of IID and manifold validation `log_l1`. `select_families` in that checkpoint is `iid,manifold`. Neither number is the harness overall 0.1–10 Hz absolute g log-L1 with the POD weight `sqrt(f / 1 Hz)`.

The training loss is a third quantity. When the basis weight is `sqrt_f`, `weighted_l1` multiplies by that weight and also doubles 0.1–0.4 Hz on windows of 1000 m and wider. The harness overall metric uses the same square-root weight and does not apply that width boost. `_objective` was not changed.

`--select-harness-sqrt` is the opt-in. It is off unless the flag is passed. It keeps the families of the active selector and replaces the value with `log_l1_overall_sqrt`, which is `score_harness.weighted_overall_g_log_l1`. `tests/test_phase6_selection.py` checks the flag default, checks that the unweighted scores ignore the new column and the blind bins, and checks that the opt-in matches the harness weight.

The early-epoch validation optimum was not retuned. `SELECT_WINDOW` and `SELECT_DELTA` are untouched. The Phase 1.7 shift audit is the reason to leave them: IID train versus IID test has adversarial AUC 0.479, while IID train versus dipping, three-layer, and the manifold pool sit at 1.00, 1.00, and 0.9996. That is the designed shift. Manifold train versus manifold validation is 0.58. Those figures are in `analysis/audit/splits.csv`.

## 6.4 Monitor

`analysis/train_monitor.py` logs one epoch when it is called. It does not call `train.py`. A row is one family and one band. The columns are the harness g log-L1, the harness harm rate, the three consistency probes, the POD floor and the realization floor from `analysis/audit/representation.csv` and `realization.csv`, and the gaps (model log-L1 minus the floor). The POD floor is the `family:val` pool. The realization floor is the median over Sobol groups. Manifold sections have no realization rows, so that gap is blank. The half-width is half of the harness record-bootstrap interval of the mean, at `gates.BOOTSTRAP_ALPHA`.

Blind rows go to `blind_sealed.csv`. `read_sealed_blind` raises `BlindSealed` before it touches the path, unless `open_blind=True`.

Proposed stop: one named consistency probe is false for `gates.CONSISTENCY_FAIL_EPOCHS` (3) epochs, or the overall-band harm rate in any family is above `gates.HARM_RATE_STOP` (5%). A rate equal to 5% does not stop. Per-band harm is logged and is not the stop statistic. Failures of different probes on different epochs do not add. The monotone probe is not in this rule; the frozen table left it unrun because no existing pair changes only CoV or only dip.

Proposed accept versus P128n128: the pooled overall validation log-L1 improves by more than that bootstrap half-width, no family's overall log-L1 is worse by more than `gates.TOST_DELTA_LOG_L1` (0.02), and the homogeneous, CoV = 0, and mirror probes pass. A delta equal to the margin does not fail. Acceptance then requires `gates.MIN_TRAINING_SEEDS` (3) training seeds. Those seeds were not launched.

`python train_monitor.py --dry-run` scores a five-record synthetic table through `score_curves` and prints the CSV columns. That command does not train.

## What this phase did not do

No new finite-element pairs. No velocity-scaling augmentation. The default loss and the default selector are unchanged. The early-stopping window was not retuned. No training run was started, and no HPC job was submitted. A new model is not part of Paper 2.
