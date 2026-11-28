# External generalization protocol

This file is the preregistration. It was written before a confirmatory retrain and before any external label is opened. The machine-readable decision is `decision.json`, produced by `external/run_local_diagnosis.py`.

## Claim and failure

The confirmatory claim, if the feasibility gate later opens, is that seed 0 lowers equal-geology 0.1–10 Hz absolute log error against the corrected local 1D outcrop column by more than 0.02, without exceeding the absolute adequacy ceiling, and that no prespecified family with at least three independent geologies is worse than its frozen margin. Failure of that gate means generalization is unsupported. It does not mean every real site fails or that the architecture can never work.

P128n128 is the second comparison and is tested only after the column comparison passes. The 9-column geometric-mean outcrop is reported at every site and is not a gate. Recorded motions are descriptive only.

Treasure Island has already been examined. It is not a blind external site. It can be called favorable inside this cohort only after the cohort is opened and its locked score sits at the favorable end.

## Cohort

A confirmatory result needs at least six, and targets eight, independent in-domain geologies. Slices from one regional velocity model count as one geology. A family supports a family-level statement only when it contains at least three geologies. Fewer than six committed geologies makes the study a descriptive Paper 3, not a rebuilt Paper 2.

Eligibility uses metadata only: interior width 500–2000 m, depth at most 250 m and 256 encoder cells, Vs and impedance inside the frozen training envelope, a flat surface, and a defensible elastic half-space. Anything else is extrapolation and stays out of the confirmatory mean. The fallback order is Mygdonian, Nice, Thorndon, Garner Valley, Parkway, then Bagneres-de-Bigorre. Grenoble and deep USGS or SCEC slices are extrapolation. Replacements follow that order, never a model score.

## Model selection

Development uses nested geology prefixes, fold-local POD and PCA, and the pure sqrt(f) overall log-L1 for both training and selection. The test split, manifold blind split, Treasure Island, and external labels stay closed. Seed 0 is the only reported model. Seeds 1 and 2 measure training variability and cannot replace it. The final fit uses a fixed epoch count on train and validation, with no checkpoint chosen by a test score.

The prespecified options are mirror augmentation, length scaling only after three Rayleigh finite-element pairs meet the identity tolerance, the current PCA/POD control, an independent depth-weighted input POD, and a supervised coupled POD whose forward pass cannot see the true residual. The learning-curve labels are:

- still falling: projected gain at twice the data exceeds 0.02 and its 95% interval excludes zero;
- flat near the floor: the gain interval lies at or below 0.02 and the floor gap is at most 10% of the 1D error;
- flat above the floor: the gain interval lies at or below 0.02 and the floor gap is larger;
- otherwise inconclusive.

An IID curve does not classify another family. Inconclusive is not a reason to widen the claim.

## Seal

External labels are encrypted. Their ciphertext hash is published before training. A custodian holds the key. The append-only ledger records the single opening, and decryption fails before that entry. Inference uses one 2000 m centered window per station at a 20 m stride. Stations without a full window are truncated and excluded. Overlapping windows are not averaged.

## Current stop

The local decision is Paper 3, descriptive. The RTX 5080 is available to this environment. No confirmatory training or external opening is authorized until at least six in-domain geometries are committed, a solver timing benchmark exists, and a custodian is named. The epoch-46 P128n128 record is unchanged.
