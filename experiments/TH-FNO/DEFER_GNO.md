# Deferral: mesh GNO + FEM K/M features

Status: **deferred** until the gated Haskell + grid-FNO residual fails after:

1. Peak-aware loss (`LOSS_PEAK_WEIGHT`)
2. RF-corner oversampling / RV-matched geostat training
3. A trained residual that still underperforms the plan success bar on Response_Variability

## Why not now

- OpenSees pipelines in seiskit do **not** export global/element \(\mathbf{K},\mathbf{M}\) to ML.
- The frequency-domain operator is \(K - \omega^2 M + i\omega C\); static \(K_{ij},M_{ij}\) edges are incomplete.
- The domain is a **regular 1 m grid**; FNO failure modes here are spectral bias / missing 1D prior, not unstructured-mesh pooling blur.
- Building a Graph Neural Operator stack would dominate engineering cost before we know whether a **1D anchor + gated residual** is sufficient.

## Escalation trigger

Revisit mesh GNO / stencil graph operators only if, after the above, RV re-score still has approximately:

- `delta_pearson` mean \(\lesssim 0.85\) and
- `delta_rel_l2` mean \(\gtrsim 0.35\)

while Haskell-only / Pretell remain strong — i.e. the residual is large and the grid FNO cannot capture lateral scattering.
