# Dipping soil thickness

The dipping atlas draws **Toro dip** and **Passeri dip**. It does not draw Toro fixed H or Passeri fixed H. Those fixed-H spectra stay in the ensemble files and in `ood_dipping_heldout_summary.csv`.

## One soil layer

NHPP layer-thickness randomization is off. The soil column is not subdivided.

- One \(V_s\) is drawn for the whole soil thickness.
- Toro uses the mid-depth lognormal draw at the case coefficient of variation. Bedrock \(V_s\) stays at the case value.
- Passeri draws one travel-time \(V_s\) for that thickness. Bedrock \(V_s\) is drawn jointly with the interface.
- The spectrum is the two-layer analytical \(|\mathrm{TF}|\), not a stack of sublayers.

## How \(H\) is changed

The case \(H\) is the thickness at the center recorder.

- Dip angle \(\theta\) is the case dip.
- The interface depth is \(H + x\tan\theta\).
- \(x\) is uniform on \([-250, 250]\,\mathrm{m}\), so the half-span of the interface is \(250\,|\tan\theta|\).
- Fixed H uses the case thickness and does not move the interface.

The plotted dip curve is the geometric mean of 200 realizations. The shaded band on the comparison panels is that geomean times \(\exp(\pm\sigma_{\ln})\).

Code: `seiskit/neural-operator/data/ood_dipping/run_toro_comparison.py` and `run_passeri_comparison.py`.
