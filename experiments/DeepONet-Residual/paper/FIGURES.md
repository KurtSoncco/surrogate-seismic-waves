# SDEE results figures

Regenerate with:

```bash
uv run python experiments/DeepONet-Residual/paper/make_figures.py
```

Each stem is a vector PDF and a 300 dpi PNG in `paper/figures/`. Column widths are the Elsevier production sizes, 90 mm and 190 mm. The graphical abstract is a fixed 13.0 cm by 5.2 cm canvas (1535 by 614 px at 300 dpi), which clears Elsevier’s 1328 px minimum width, and is also written as a TIFF. Text is Arial, registered from the Windows font folder when the figures are built under WSL. Color is never the only cue: ranking markers and transfer-function linestyles differ, and domain colors are not mixed with method colors on the same axes.

Frozen nested tests shown here are IID 150 and dipping 144. Validation versus test uses the held-out packs (IID 150/150, dipping 144/144).

## Fig. 1 — `fig01_training`

Fine-tune of `M700_gino.pt` stored in `M7680_gino_rebal_ft.pt` (82 epochs). The curves are the stored training loss and the pooled validation scores.

- **a.** Training Smooth L1 on \(R\) falls from 0.059 to 0.035, while pooled validation Smooth L1 stays near 0.085.
- **b.** Pooled validation relative \(L_2\) of \(|\mathrm{TF}|\). The marker is the shipped epoch. The loss keeps falling after that epoch and the validation score does not.

## Fig. 2 — `fig02_residual`

Same nested tests. Relative \(L_2\) of \(|\mathrm{TF}|\), lower is better.

- **a.** LOGLO is slightly ahead in family (0.34 versus GINO 0.35) and then worse than the 1D base case on dipping (0.93 versus 0.60). Residual GINO stays under the 1D base case on both slices (0.35, 0.32).

## Fig. 3 — `fig03_metrics`

- **a–d.** Frozen-test relative \(L_2\), Pearson of \(|\mathrm{TF}|\), leftover \(R^2\), and harm rate: IID 0.35 / 0.93 / 0.61 / 2.0%; dipping 0.32 / 0.90 / 0.70 / 0.7%.
- **e.** Central-recorder Pearson on validation and test. IID medians are both 0.94. Dipping medians are 0.92 (validation) and 0.93 (test). The larger val+test cloud is not a different result.

## Fig. 4 — `fig04_sota`

Points in **a–b** are medians with cluster-bootstrap intervals from `ood_method_ranking.csv`. **c–d** are IID means by band (0.1–0.5, 0.5–2, 2–10 Hz), the same averages as the published band table.

- **a–b.** GINO leads Pearson on IID and dipping (median 0.94 and 0.93). Pretell median is close on IID and behind on dipping. Toro, Passeri, and Dmult sit with the 1D base case.
- **c–d.** On IID, Pretell median wins the low band. GINO leads the mid and high bands (Pearson 0.94 and 0.79).

## Fig. 5 — `fig05_iid`

Four curves only: OpenSees 2-D, GINO, the 1D base case, and Pretell median. Cases are the nested IID test sample nearest the median Pearson, and a CoV above the 75th percentile near the median Pearson of that subset.

- **a–c.** Typical case, Pearson 0.94, CoV 0.23, \(H=36\) m. GINO follows OpenSees through the fundamental and the first overtones.
- **d–f.** Rougher case, Pearson 0.90, CoV 0.29, \(H=88\) m. The leftover spikes are larger, and \(\hat R\) returns only part of them.

## Fig. 6 — `fig06_ood`

Dipping test cases only. The column is the nominal 1-layer profile at the central recorder.

- **a–c.** Typical case, Pearson 0.92, CoV 0.17, \(H=36\) m.
- **d–f.** High-CoV case, Pearson 0.83, CoV 0.23, \(H=55\) m. Above about 2 Hz, GINO stays on OpenSees where the 1D base case and Pretell median fall off.

## Fig. 7 — `fig07_leftover`

Trough-windowed central recorder. The line is the cached OLS \(\hat R=a+bR\) (`gino_bias/M7680_gino_rebal_ft/summary.json`), with the cluster-bootstrap interval.

- **a.** IID, \(b=0.58\) [0.42, 0.77].
- **b.** Dipping, \(b=0.70\) [0.67, 0.73].

The slope stays below the 1:1 line: GINO returns a fraction of \(R=\mathrm{TF}_{2D}-\mathrm{TF}_{1D}\) and does not overshoot.

## Graphical abstract — `graphical_abstract`

Read left to right. Elsevier shows it 200 px high, so the type is 8 pt or larger and there is no legend row or table.

- **Site.** The \(V_s\) section of the dipping test case nearest the median Pearson (the typical case of Fig. 6), with the 21 surface recorders and the nominal 1-D column under the central one.
- **Leftover.** \(|\mathrm{TF}|\) at the central recorder for OpenSees 2-D, the 1D base case, and GINO. The blue band between the first two is \(R\).
- **Held-out tests.** Median Pearson of \(|\mathrm{TF}|\) from `ood_method_ranking.csv`, IID ("Flat") to dipping: GINO 0.94 to 0.93, Pretell median 0.93 to 0.76, 1D base case 0.77 to 0.69.

The abstract uses its own colors (blue GINO, orange Pretell, grey 1D base case, sand-to-rock section); Figs. 1–7 keep `METHOD_STYLE`.

Dipping is a held-out test slice, not an unseen geometry: the M7680 mix trains on the dipping train split.
