# OOD TF head-to-head: LOGLO-POD vs ResUNet DeepONet (R_nom)

**1920** OOD cases (960 dipping + 960 three_layer), **full 1000 frequencies**.
Ground truth from cached `ood_scores_full7680` OpenSees TFs.

| Campaign | Model | mean rel L2 ↓ | mean Pearson_f ↑ | mean R²_TF ↑ | win rate (rel L2) |
|----------|-------|--------------:|-----------------:|-------------:|------------------:|
| **dipping** | DeepONet (TF₁D+R̂) | **0.638** | **0.643** | **0.089** | **93.5%** |
| dipping | LOGLO-POD full7680 | 0.928 | −0.021 | −0.870 | 6.5% |
| dipping | TF₁D_nom only | 0.580 | 0.635 | 0.269 | — |
| **three_layer** | **LOGLO-POD full7680** | **0.807** | **0.360** | **0.072** | **92.1%** |
| three_layer | DeepONet (TF₁D+R̂) | 0.955 | 0.021 | −0.295 | 7.9% |
| three_layer | TF₁D_nom only | 0.956 | −0.002 | −0.298 | — |
| **combined** | **DeepONet (TF₁D+R̂)** | **0.796** | **0.332** | **−0.103** | **50.7%** |
| combined | LOGLO-POD full7680 | 0.868 | 0.170 | −0.399 | 49.3% |
| combined | TF₁D_nom only | 0.768 | 0.316 | −0.015 | — |

## Pearson head-to-head (Pearson_f: mean over recorders, per spectrum)

| Campaign | LOGLO wins | DeepONet wins | vs TF₁D: LOGLO | vs TF₁D: DeepONet |
|----------|------------|---------------|----------------|-------------------|
| dipping | 2.0% | **98.0%** | 0.7% | 58.1% |
| three_layer | **94.6%** | 5.4% | **95.2%** | 66.8% |
| combined | 48.3% | **51.7%** | 48.0% | 62.4% |

In-dist reference (1152 test): LOGLO wins Pearson **93.7%** vs DeepONet; both beat TF₁D on ~97–99% of cases.

## Verdict

**LOGLO does not win OOD overall.** DeepONet edges combined (50.7% win rate, lower mean rel L2) and dominates dipping (93.5%). LOGLO wins three_layer strongly (92.1%) where DeepONet collapses to ~TF₁D_nom baseline.

This reverses the in-distribution result (LOGLO 0.242 vs DeepONet 0.356, 95.5% win rate on 1152 test samples).

## Prior LOGLO-only OOD (per-recorder mean rel L2)

From `ood_scores_full7680/overall_summary.json` (different aggregation than full-TF rel L2 here):

| Campaign | LOGLO-only rel_l2_mean | This run LOGLO rel_l2 (full TF) |
|----------|----------------------:|--------------------------------:|
| dipping | 0.915 | 0.928 |
| three_layer | 0.768 | 0.807 |

LOGLO Pearson_f means match exactly (dipping −0.021, three_layer 0.360), confirming consistent LOGLO predictions vs prior scoring.

**In-dist comparison:** [`compare_tf_loglo_vs_deeponet/README.md`](../compare_tf_loglo_vs_deeponet/README.md) (1152 test; LOGLO wins both rel L2 and Pearson).

## Artifacts

- `summary.json` — full aggregates + per-case metrics
- Lambda log: `~/compare_tf_ood.log`
- Script: `experiments/DeepONet-Residual/compare_tf_ood_loglo_vs_deeponet.py`
- Launcher: `experiments/DeepONet-Residual/run_ood_compare_lambda.sh`
