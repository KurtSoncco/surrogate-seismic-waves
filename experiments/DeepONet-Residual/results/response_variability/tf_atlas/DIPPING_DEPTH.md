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

## Fixed \(\theta\) is a uniform depth

With \(\theta\) locked to the case dip, \(x\) uniform on \([-250, 250]\,\mathrm{m}\) maps to a uniform interface depth on \([H - 250|\tan\theta|,\, H + 250|\tan\theta|]\). Half-widths on the 32 Sobol points run from \(0.07\,\mathrm{m}\) to \(12.8\,\mathrm{m}\) (median \(6.4\,\mathrm{m}\)).

On every point, 8000 Toro draws and 8000 Passeri draws stay inside that interval (clipped fraction 0). The Kolmogorov–Smirnov distance to the uniform has median \(0.010\) (Toro) and \(0.009\) (Passeri). The largest distance is \(0.015\), which is the \(5\%\) critical value \(1.36/\sqrt{8000}\). That is sampling noise around a uniform, for both models. Table: `seiskit/neural-operator/data/ood_dipping/figures/fixed_sobol_depth_ks.csv`.

The paper figures lock \(\theta\) to the Sobol point whose \(|\theta|\) is closest to the median, and draw only \(x\). `figures/toro_passeri_dip_depth.png` and `figures/toro_passeri_dip_depth_all_sobol.png` are that case: the dip histogram sits on the uniform density, and the Toro and Passeri profiles use the same \(V_{s1}\), \(H\), CoV, and \(V_{s2}\). `figures/toro_passeri_dip_depth_fixed_sobol.png` also shows the shallow and steep Sobol points, plus the angle-mixture histograms those paper figures used to draw.

The scored geomeans use the fixed-\(\theta\) uniform. Rebuilding the Toro dip geomean with the comparison seeds matches `toro_comparison/ensembles.h5` (relative \(L_1 = 0\)).

Swapping the locked angle for \(\theta\sim\mathrm{Unif}[-3^\circ,3^\circ]\), with the same \(V_s\) seeds, moves the geomean by a median relative \(L_1\) of \(0.022\) (Toro) and \(0.024\) (Passeri). Against the cached center 2D \(|\mathrm{TF}|\), replicate-median Pearson goes from \(0.662\) to \(0.672\) for Toro (median \(\Delta r = +0.005\)) and from \(0.689\) to \(0.696\) for Passeri (median \(\Delta r = +0.004\)). Split at the median \(|\theta|\):

- Shallow half: median \(\Delta r = -0.012\) (Toro) and \(-0.006\) (Passeri). The mixture adds depth range those dips do not have.
- Steep half: median \(\Delta r = +0.015\) (Toro) and \(+0.016\) (Passeri). The mixture pulls those cases back toward the center thickness, closer to the fixed-\(H\) curve.

The published fixed-\(H\) versus dip gap is larger than this swap (Toro median \(\Delta r = -0.015\), relative \(L_1 = 0.031\)) and it is computed at the case dip. Per-point scores: `figures/fixed_sobol_tf_delta.csv`.
