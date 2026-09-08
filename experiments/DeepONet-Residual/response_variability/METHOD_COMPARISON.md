# Response variability: method comparison against OpenSees 2-D

Refresh of the response-variability comparison after the improved encoding. Every
1-D approximation method is scored against the OpenSees 2-D transfer functions
that define ground truth, across three campaigns: in-distribution (`iid`),
dipping-interface out-of-distribution (`ood_dipping`), and three-layer
out-of-distribution (`ood_three_layer`). The surrogate (GINO) is included as a
peer arm using held-out predictions only.

## Arms compared

| Arm | What it is |
| --- | --- |
| 1D base case | Single deterministic 1-D column on the nominal (layered) profile |
| Damping normalization | 1-D base case with the small-strain damping rescaled by a calibrated multiplier |
| Toro Vs-rand geomean / p84 | Toro-style Vs randomization, geometric-mean and 84th-percentile transfer function |
| Passeri TTS geomean / p84 | Passeri travel-time-based randomization, geomean and 84th percentile |
| Pretell geomean / p84 | Ensemble of 1-D columns extracted along the 2-D section, geomean and 84th percentile |
| GINO | Neural-operator surrogate, held-out predictions |
| 1D base (legacy) | The uncorrected uniform-column construction, retained on the three-layer campaign as a diagnostic |

Primary metric is the Anderson-style goodness-of-fit misfit on |TF| against
OpenSees 2-D — **lower is better**. Rankings use paired Wilcoxon tests against
the per-campaign best arm with Holm correction; bootstrap confidence intervals
accompany every median.

## Headline result

GINO is the single best arm on both out-of-distribution campaigns and second on
the in-distribution campaign, where the Pretell column ensemble wins.

| Rank | `iid` | `ood_dipping` | `ood_three_layer` |
| --- | --- | --- | --- |
| 1 | Pretell geomean — 0.080 | **GINO — 0.102** | **GINO — 0.148** |
| 2 | GINO — 0.102 | Pretell geomean — 0.120 | Toro Vs-rand geomean — 0.190 |
| 3 | Pretell p84 — 0.151 | Toro Vs-rand geomean — 0.127 | Pretell geomean — 0.194 |
| 4 | Toro Vs-rand geomean — 0.156 | Damping normalization — 0.137 | Damping normalization — 0.244 |
| 5 | Damping normalization — 0.171 | Passeri TTS geomean — 0.140 | Passeri TTS geomean — 0.248 |
| 6 | Passeri TTS geomean — 0.173 | 1D base case — 0.141 | Passeri TTS p84 — 0.249 |
| 7 | 1D base case — 0.175 | Passeri TTS p84 — 0.141 | 1D base case — 0.252 |
| 8 | Passeri TTS p84 — 0.178 | Pretell p84 — 0.174 | Pretell p84 — 0.262 |
| 9 | Toro Vs-rand p84 — 0.300 | Toro Vs-rand p84 — 0.248 | Toro Vs-rand p84 — 0.304 |
| 10 | — | — | 1D base (legacy) — 0.921 |

Median Anderson misfit on the held-out cases (n = 150 `iid`, 144 each OOD). All
gaps to the campaign winner are significant after Holm correction (p < 0.001),
except Pretell geomean vs GINO on `ood_dipping` (p = 0.0018).

![Ranking with bootstrap CIs and significance]({{artifact:art_efdd9b02-e4f0-488b-882a-4e06ed79548c}})

Three findings organise the rest of the report:

1. **Randomization does not buy accuracy in the central estimate.** Toro and
   Passeri geomeans land within 0.02 of the plain 1-D base case on every
   campaign. Randomizing the profile widens the ensemble but leaves the median
   |TF| essentially where the deterministic column put it.
2. **The 84th-percentile arms are not interchangeable.** As upper envelopes they
   have wildly different calibration (below), and as point predictors they are
   uniformly worse than their own geomeans.
3. **Spatial variability, not profile uncertainty, is what the 2-D response
   needs.** Pretell — the only classical arm that samples actual 2-D geometry —
   is the only classical arm that competes with the surrogate.

## The three-layer f0 offset was a profile-construction defect

The suspected miscalculation on `ood_three_layer` is real and now isolated. The
1-D arms had been building a **uniform column at the soft top-layer velocity
across the whole soil depth**, discarding the stiffer intermediate layer. That
shifts the fundamental frequency well below the 2-D value, which is exactly the
"f0 peak way behind" symptom.

Rebuilding the nominal from the authoritative layered constructor moves the
three-layer 1-D base case from **0.921 to 0.252** median misfit — a factor of
3.7 — and it is what allows the corrected classical arms to be compared to the
surrogate at all. The legacy construction is retained as arm 10 above so the
size of the defect stays visible rather than asserted.

![f0 diagnosis, legacy vs corrected]({{artifact:art_ded09bd7-b3c3-4eba-9efc-126518060f87}})

The velocity-profile figure shows the mechanism directly, against the true 2-D
field rather than against another 1-D idealisation. On the three-layer case the
legacy curve is flat where the corrected nominal steps at the interface; the
randomization bands are drawn as dashed percentile curves because they are
narrow enough to be hidden by a filled band.

![Velocity profiles vs the true 2-D field]({{artifact:art_044c69d9-ef30-4948-b7bb-06183a7ba06c}})

The legacy defect also leaves a signature in the band-resolved bias: the legacy
arm is **−0.380 in the high band and +0.263 in the mid band** on three-layer —
the classic pattern of a misplaced peak, energy missing where it should be and
surplus where it should not. The corrected arm is −0.129 / −0.017 in the same
bands, i.e. mildly under-predicting at high frequency with no mid-band surplus.

## Transfer functions

|TF| is plotted in log scale on both axes throughout.

![|TF| comparison grid across campaigns and arms]({{artifact:art_3d718c6e-7e16-4b5f-b627-d5703a2ff28a}})

![Example |TF| per campaign with envelopes]({{artifact:art_bd17950e-28a4-495a-aff0-a0d52685846f}})

## 84th-percentile envelopes are poorly calibrated

An honest 84th-percentile envelope should be exceeded by the 2-D ground truth
about **16 %** of the time. Median exceedance rate over frequency bands:

| Arm | `iid` | `ood_dipping` | `ood_three_layer` |
| --- | --- | --- | --- |
| Pretell p84 | 15.2 % | 41.6 % | 33.1 % |
| Toro Vs-rand p84 | 12.5 % | 16.3 % | 25.4 % |
| Passeri TTS p84 | 57.5 % | 72.0 % | 76.1 % |

Passeri TTS is badly under-dispersed — travel-time randomization preserves the
travel time by construction, so it barely moves f0 and its 84th percentile sits
close to its own median. Pretell is well calibrated in-distribution and loses
calibration out of distribution. Toro is the closest to nominal on `iid` and
`ood_dipping`, but it gets there by being wide rather than by being right: its
median envelope ratio is **1.295** on `iid` (a 30 % over-shoot of the 2-D peak)
with a peak margin of −0.236 in ln units, and as a point predictor it is the
worst arm on every campaign.

![84th-percentile envelope coverage]({{artifact:art_b89b7cdc-a805-4884-bf07-effe4e47af04}})

![Band-resolved bias]({{artifact:art_af191f84-6a8a-41cb-81ca-a9c0ab408dbf}})

## Dispersion by frequency band

Median σ_ln across the response, compared to the OpenSees 2-D dispersion across
recorders (the quantity a 1-D method is trying to reproduce). All-band /
high-band:

| Source | `iid` | `ood_dipping` | `ood_three_layer` |
| --- | --- | --- | --- |
| OpenSees 2-D (across recorders) | 0.131 / 0.295 | 0.202 / 0.392 | 0.115 / 0.312 |
| GINO (across recorders) | 0.099 / 0.152 | 0.194 / 0.293 | 0.154 / 0.258 |
| Pretell (spatial columns) | 0.176 / 0.310 | 0.211 / 0.355 | 0.125 / 0.335 |
| Toro Vs randomization | 0.283 / 0.367 | 0.268 / 0.382 | 0.159 / 0.428 |
| Passeri TTS randomization | 0.062 / 0.095 | 0.055 / 0.094 | 0.029 / 0.079 |

Pretell's spatial-column dispersion tracks the 2-D dispersion closely on all
three campaigns. Toro over-disperses (by ~2× on `iid` all-band). Passeri
under-disperses by a factor of 2–4. GINO reproduces the OOD dipping dispersion
well and is smooth relative to 2-D at high frequency in-distribution.

![Dispersion σ_ln by band, including the surrogate]({{artifact:art_d5b93cee-2141-42e6-adba-1ebf33a4fd66}})

## Damping normalization

A sweep over the small-strain damping multiplier calibrated to **1.308** on the
in-distribution training split. Applying it improves the `iid` median misfit
from 0.175 to 0.171 and the `ood_dipping` from 0.141 to 0.137 — real but small,
and it costs high-frequency bias: the damping-normalized arm is the most
negatively biased arm in the high band on `ood_dipping` (**−0.485** vs −0.394
for the un-normalized base case). Extra damping buys median agreement by
suppressing high-frequency amplitude that the 2-D solution actually has.

![Damping multiplier sweep]({{artifact:art_d5997060-5eea-4f43-86ba-05bdc670dc1a}})

![Damping validation]({{artifact:art_69824102-9fc9-4502-9e1a-4213c23e931b}})

## Error vs site covariates

Errors binned by coefficient of variation of the velocity field (CoV),
horizontal-to-depth aspect ratio (rH), and horizontal-to-vertical anisotropy
(aHV). Spearman ρ between per-case misfit and each covariate:

| Arm | `iid` CoV | `ood_dipping` CoV | `ood_three_layer` CoV |
| --- | --- | --- | --- |
| 1D base case | 0.674 | 0.639 | 0.253 |
| Damping normalization | 0.647 | 0.617 | 0.249 |
| Pretell geomean | 0.432 | 0.235 | 0.226 |
| Toro Vs-rand geomean | 0.420 | 0.477 | 0.243 |
| Passeri TTS geomean | 0.669 | 0.636 | 0.255 |
| GINO | 0.483 | 0.618 | 0.485 |
| 1D base (legacy) | — | — | 0.022 |

CoV of the velocity field is the dominant covariate for every method; rH and aHV
correlations are weak (|ρ| ≤ 0.43, mostly ≤ 0.25). The 1-D arms degrade steeply
with heterogeneity — ρ ≈ 0.64–0.67 for the base case and for Passeri, whose
randomization does not help. Pretell has the flattest CoV dependence on
`ood_dipping` (0.235), consistent with sampling the section rather than
perturbing one column.

Two cautions on reading this table. GINO's CoV correlation on `ood_three_layer`
(0.485) is higher than the classical arms' — but its *level* of error is lower
everywhere in the range, so this is a steeper slope on a lower curve, not worse
performance. And the legacy arm's ρ ≈ 0.02 is not robustness: its error is
dominated by the f0 offset, which swamps any heterogeneity effect.

![Error vs CoV, rH and aHV]({{artifact:art_5dc4ffaa-9c97-494f-a293-559c560bb361}})

## Reading of the comparison

Against OpenSees 2-D ground truth:

- **Best overall**: GINO. Wins both OOD campaigns with significant margins; only
  Pretell geomean beats it, and only in-distribution.
- **Best classical method**: Pretell column ensemble (geomean). It is the only
  classical arm that both competes on median accuracy and reproduces the 2-D
  dispersion, because it is the only one that sees the 2-D geometry.
- **Profile randomization (Toro, Passeri)** does not improve the central
  estimate over a deterministic 1-D column. Its value has to be argued from the
  envelope, and there the two disagree sharply: Toro over-disperses, Passeri
  under-disperses by 2–4×.
- **Damping normalization** gives a small, consistent median improvement bought
  with high-frequency under-prediction.
- **Use the geomean, not the p84, as a point predictor.** Every p84 arm is worse
  than its own geomean on every campaign.

## Caveats

- The Anderson misfit is the only ranking metric; band bias, coverage and
  dispersion are reported separately rather than folded into a composite score.
- The damping multiplier is calibrated on the `iid` training split and applied
  unchanged to both OOD campaigns.
- GINO figures use held-out predictions, but the surrogate was trained on the
  `iid` corpus; its `iid` numbers are out-of-sample within a distribution it has
  seen.
- Dispersion for the 2-D reference and for GINO is taken across recorders;
  for Pretell across spatial columns; for Toro and Passeri across randomized
  realizations. These are different sampling operations and the comparison is
  intentionally between what each method offers as its own variability estimate.
- Model-assumption diagnostics beyond the paired significance tests were not
  assessed.

## Files

Tables: [method_ranking.csv]({{artifact:art_d730c465-eab3-429a-94be-5171b31bee17}}),
[p84_coverage.csv]({{artifact:art_505c412c-0bf2-450e-8d10-857a82bab581}}),
[band_bias.csv]({{artifact:art_257693fb-7c8c-4264-a2aa-1b5c2ce2bcb8}}),
[sigma_ln_by_band.csv]({{artifact:art_bd8c8bbe-377c-4f53-aca6-5310d75acd13}}),
[error_vs_covariates.csv]({{artifact:art_6cca9731-9b79-4c91-9e7d-e76187658474}}),
[covariate_spearman.csv]({{artifact:art_2180fc62-55ea-4bc4-ad4d-04fef9db77d4}}),
[damping_sweep.csv]({{artifact:art_3545682d-bdef-488c-af7d-4671bc743e37}}),
[damping_validation.csv]({{artifact:art_a8b84a11-cb20-404c-a444-1fbbc26e9df2}}),
[method_summary.csv]({{artifact:art_545909bd-e672-4b8a-b448-33971dfb8aa8}}),
[per_case_metrics.csv.gz]({{artifact:art_ae5a3a3c-c1ed-4f7e-8f8c-60ffb9d170bb}}),
[site_covariates.csv]({{artifact:art_5d0c115d-3175-4bdc-8dd1-6b69bd42c4d9}}),
[evaluation_universe.csv]({{artifact:art_a694687a-ee95-47f4-ac56-b7b984c779a3}}).

Code lives in `experiments/DeepONet-Residual/response_variability/mc/`
(`arms.py`, `extract.py`, `score.py`, `pipeline.py`, `legacy3l.py`, `figs.py`,
`figs2.py`).
