# Spatial leftover vs inflated TF Pearson

Pack-only diagnostic: nested IID and dipping. Pearson of $|\mathrm{TF}|(f)$ is affine-invariant, so a 1-D resonance can look strong even when the predictor has no lateral field.

## 1. Is the 2-D field too smooth?

| Domain | OpenSees spatial $\sigma_{\ln}$ | Edge vs center Pearson | Adjacent Pearson |
| --- | ---: | ---: | ---: |
| iid | 0.131 [0.101, 0.171] | 0.908 [0.823, 0.960] | 0.996 [0.992, 0.998] |
| dipping | 0.213 [0.153, 0.262] | 0.668 [0.477, 0.830] | 0.985 [0.970, 0.992] |

IID stations share almost the same spectrum (adjacent Pearson near 1). That is lateral smoothness of *shape*, not a missing random field: spatial $\sigma_{\ln}$ is positive, and CoV quartiles below still move it. Dipping is the geometry leftover (edge vs center drops).

CoV quartile of OpenSees spatial $\sigma_{\ln}$ / 1-D Pearson:

- **iid:** Q1: $\sigma_{\ln}$ 0.091 [0.076, 0.105], 1-D Pearson 0.887 [0.837, 0.930]; Q2: $\sigma_{\ln}$ 0.118 [0.109, 0.138], 1-D Pearson 0.806 [0.733, 0.870]; Q3: $\sigma_{\ln}$ 0.152 [0.130, 0.190], 1-D Pearson 0.704 [0.604, 0.771]; Q4: $\sigma_{\ln}$ 0.188 [0.152, 0.213], 1-D Pearson 0.666 [0.519, 0.751].
- **dipping:** Q1: $\sigma_{\ln}$ 0.121 [0.109, 0.197], 1-D Pearson 0.811 [0.684, 0.881]; Q2: $\sigma_{\ln}$ 0.228 [0.169, 0.264], 1-D Pearson 0.725 [0.645, 0.774]; Q3: $\sigma_{\ln}$ 0.225 [0.176, 0.266], 1-D Pearson 0.678 [0.656, 0.766]; Q4: $\sigma_{\ln}$ 0.224 [0.192, 0.290], 1-D Pearson 0.531 [0.431, 0.610].

## 2. Why SOTA looks good

| Domain | Arm | Central Pearson | Array-mean Pearson | Spatial $\sigma_{\ln}$ | Spatial-pattern Pearson |
| --- | --- | ---: | ---: | ---: | ---: |
| iid | 1D Base Case | 0.772 [0.659, 0.876] | 0.748 [0.659, 0.848] | 0.000 [0.000, 0.000] | — |
| iid | Toro Vs | 0.844 [0.782, 0.890] | 0.838 [0.796, 0.868] | 0.000 [0.000, 0.000] | — |
| iid | Pretell median | 0.926 [0.878, 0.952] | 0.921 [0.881, 0.945] | 0.000 [0.000, 0.000] | — |
| iid | GINO | 0.943 [0.910, 0.970] | 0.938 [0.906, 0.961] | 0.100 [0.077, 0.131] | 0.246 [0.145, 0.297] |
| dipping | 1D Base Case | 0.690 [0.596, 0.798] | 0.627 [0.538, 0.734] | 0.000 [0.000, 0.000] | — |
| dipping | Toro Vs | 0.731 [0.642, 0.811] | 0.695 [0.597, 0.771] | 0.000 [0.000, 0.000] | — |
| dipping | Pretell median | 0.758 [0.681, 0.857] | 0.709 [0.624, 0.811] | 0.000 [0.000, 0.000] | — |
| dipping | GINO | 0.928 [0.883, 0.955] | 0.916 [0.874, 0.942] | 0.194 [0.157, 0.237] | 0.452 [0.348, 0.566] |

- **1-D Base Case** already owns the resonance; array-mean Pearson barely moves on IID because every station looks like the same column.
- **Pretell median** is still 1-D wave physics on the *true* 2-D $V_s$ strip (200-column geomean). That lifts IID shape; it does not invent dipping/scattering, so the OOD gap vs GINO remains.
- **Toro** is a broadcast 1-D geomean (spatial $\sigma_{\ln}=0$, spatial-pattern Pearson undefined).

## 3. GINO is good on TF Pearson because it starts from 1-D

GINO reconstructs $\widehat{\mathrm{TF}}=\mathrm{TF}_{1D}+\hat R$. TF Pearson inherits the 1-D backbone. Leftover $R$ is the part that is actually 2-D:

| Domain | TF Pearson (GINO) | Pearson of $R$ | $R^2$ pooled | $\|R\|/\|\mathrm{TF}\|$ | Spatial-pattern Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| iid | 0.943 [0.910, 0.970] | 0.874 [0.798, 0.921] | 0.557 | 0.520 [0.372, 0.623] | 0.246 [0.145, 0.297] |
| dipping | 0.928 [0.883, 0.955] | 0.866 [0.817, 0.911] | 0.726 | 0.535 [0.442, 0.620] | 0.452 [0.348, 0.566] |

If TF Pearson is high while Pearson of $R$ and spatial-pattern Pearson are modest, GINO is under-correcting a real leftover — not saturating 2-D. The OOD Pearson gap vs Pretell is the part that is *not* metric inflation: geometry the 1-D arms cannot invent.

