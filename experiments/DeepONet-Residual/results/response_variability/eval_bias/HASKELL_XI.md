# Haskell ξ vs OpenSees soil ζ

Shipped 1-D Base Case is Thomson–Haskell with **fixed ξ=0.05**. OpenSees 2-D Rayleigh is matched to H5 `Damping_zeta` (Q–Vs `global_avg`) at $(f_0, 10\,\mathrm{Hz})$. The fair constant-$Q$ Haskell uses that soil-mean ζ, not 0.05. Rayleigh $\zeta(f)$ still differs away from the two match frequencies; this check is only the ξ-**value** mismatch. Both arms are hysteretic Haskell, so they cannot see damping *model shape* (Rayleigh vs constant-$Q$).

**Anderson** here is a Gaussian-weighted $L_1$ on $\ln|\mathrm{TF}|$, centered at the OpenSees 2-D peak frequency (width 1.5 Hz); lower is better.

| Domain | Soil ζ | Bedrock ζ | Pack nom vs ξ=0.05 max rel. err |
| --- | ---: | ---: | ---: |
| iid | 0.041 [0.037, 0.044] | 0.004 [0.004, 0.004] | 3.74e-08 |
| dipping | 0.039 [0.035, 0.042] | 0.008 [0.007, 0.009] | 3.73e-08 |

| Domain | Arm | Pearson vs 2-D | Anderson | $\Delta\ln A_{\mathrm{peak}}$ |
| --- | --- | ---: | ---: | ---: |
| iid | 1D Haskell ξ=0.05 | 0.772 [0.659, 0.876] | 0.165 [0.112, 0.238] | -0.086 [-0.334, 0.230] |
| iid | 1D Haskell ξ_soil | 0.741 [0.616, 0.861] | 0.169 [0.114, 0.244] | 0.146 [-0.139, 0.450] |
| dipping | 1D Haskell ξ=0.05 | 0.690 [0.596, 0.798] | 0.172 [0.114, 0.937] | 0.092 [-0.158, 0.290] |
| dipping | 1D Haskell ξ_soil | 0.674 [0.570, 0.789] | 0.178 [0.118, 0.873] | 0.366 [0.074, 0.548] |

The **0.05 vs soil-ζ peak** figures below are the median of *paired per-realization* $\ln(A_{0.05}/A_{\zeta})$, not the arithmetic difference of the two marginal $\Delta\ln A$ vs 2-D medians in the table (those marginals are wide and correlated, so subtracting them does not recover the paired number).

## Reading

- Soil ζ sits uniformly *below* 0.05, so switching to soil ζ is *less* damping and must raise peak amplitude. It does: paired $\Delta\ln A$ is negative (ξ=0.05 shorter than soil-ζ) in both domains, and the domain with the larger ζ-gap also has the larger peak-amplitude gap.
- Correcting to the physically correct damping makes Pearson vs 2-D **worse**, not better. Nominal ξ=0.05 was mildly *flattering* the 1-D score by accidentally mimicking part of the real 2-D peak-broadening. The ~0.77 / ~0.69 gap is therefore not a damping-*value* artifact. It is not yet a damping-*model-shape* check — that needs a literal OpenSees 1-D Rayleigh arm (`OPENSEES1D.md`).
- Residual GINO is trained on the 0.05 nom; this table does not retrain.

- **iid:** soil ζ 0.041 [0.037, 0.044] vs 0.05; Pearson 0.772 (0.05) → 0.741 (soil ζ); marginal $\Delta\ln A$ vs 2-D -0.086 vs 0.146; paired 0.05 vs soil-ζ peak -0.198 [-0.295, -0.135].
- **dipping:** soil ζ 0.039 [0.035, 0.042] vs 0.05; Pearson 0.690 (0.05) → 0.674 (soil ζ); marginal $\Delta\ln A$ vs 2-D 0.092 vs 0.366; paired 0.05 vs soil-ζ peak -0.254 [-0.348, -0.174].

## Dipping Anderson tail

Dipping Anderson IQR upper bound is ~5× the median and barely moves between damping arms. The same samples dominate both columns. Pooled $A_{\mathrm{peak}}$ over 0.1–10 Hz is grabbing a high-frequency lobe on the 2-D spectrum ($f_{\mathrm{ref}}\approx 8$–$9$ Hz vs $f_0=V_{s1}/4H$), so the tail is a metric / high-$f$ issue — not a ξ-value effect. It is also not a single geometry: $\Delta f_{\mathrm{peak}}$ q25 is already $\approx -5.5$ Hz, so **at least a quarter** of dipping 2-D spectra have their 0.1–10 Hz global max far above $f_0$. Mode-windowed peaks in `OPENSEES1D.md` avoid this pooled argmax.

| sample | $V_{s1}$ | $H$ | $f_0$ | $f_{\mathrm{peak}}$ 2-D | $\Delta f_{\mathrm{peak}}$ | Anderson 0.05 | Anderson soil-ζ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 118 | 115.7 | 32 | 0.90 | 8.55 | -7.64 | 1.689 | 1.611 |
| 87 | 177.5 | 30 | 1.48 | 8.83 | -7.35 | 1.504 | 1.419 |
| 25 | 145.7 | 58 | 0.63 | 8.99 | -8.36 | 1.468 | 1.332 |
| 59 | 232.0 | 29 | 2.00 | 7.80 | -5.80 | 1.438 | 1.425 |
| 24 | 177.5 | 30 | 1.48 | 9.12 | -7.64 | 1.431 | 1.391 |
| 110 | 194.6 | 53 | 0.92 | 7.65 | -6.74 | 1.396 | 1.255 |
| 63 | 145.7 | 58 | 0.63 | 8.71 | -8.08 | 1.387 | 1.256 |
| 135 | 177.5 | 30 | 1.48 | 9.16 | -7.68 | 1.370 | 1.289 |
