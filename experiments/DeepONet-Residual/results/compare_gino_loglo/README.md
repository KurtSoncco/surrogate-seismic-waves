# Nested-test comparison: LOGLO-POD full vs residual GINO rebal FT

Same **seed-42 nested tests** used to ship GINO: IID 150, dipping 144, three-layer 144.
This is the only head-to-head that matches the leftover recipe. Older
`compare_tf_loglo_vs_deeponet*` folders scored **ResUNet** on LOGLO’s 1152-row
prefix test and are deleted.

Pooled relative L2 on |TF| (lower is better):

| Model | IID | dipping | three-layer |
|-------|----:|--------:|------------:|
| 1D Haskell nom | 0.565 | 0.601 | 0.730 |
| **LOGLO-POD full7680** | **0.338** | 0.930 | 0.843 |
| **GINO rebal FT** | 0.353 | **0.322** | **0.525** |

GINO Pearson on |TF| (mean over frequency): IID **0.926**, dipping **0.903**, three-layer **0.869**. Harm rates 2.0% / 0.7% / 6.2%.

**Read:** LOGLO remains the better in-family amplitude fit. It does not carry a 1D prior, so layered/dipping geometry collapses. GINO is a leftover on Haskell nom: slightly behind LOGLO on IID, clearly ahead on both OOD slices, and ahead of 1D Haskell everywhere.

Sources: `arch_train/M7680_gino_rebal_ft.json` and a LOGLO forward pass on the same indices (`summary.json`).
