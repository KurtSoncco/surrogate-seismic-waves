# Box paths for the compatibility audit

The mount is `/mnt/box`. The campaign root used by this repo is

`/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data`

`/mnt/box_lab` is not mounted. Nothing below was copied into the repo. “Readable” means `h5py` opened one file and returned its top-level keys.

`data/vardomain/manifold_labels/meta.json` still points at the manifold H5 directory below. That directory is not in the local tree (`data/vardomain/manifold150/h5` is absent). The local manifold cache is the per-section `data/vardomain/cache/sections/manifold__man_*.npz` files.

## Present on the mount

| Path | What it is | H5s | Readable | Local copy |
| --- | --- | --- | --- | --- |
| `.../vardomain/manifold150/h5` | Manifold OpenSees sections. Siblings in `manifold150/` are `manifest.csv` (312 rows), `design.csv`, `retry.csv`. `vardomain/` has no other campaign. | 301 (`man_0000.h5` …) | yes (`Vs_realization_2D`, `Damping_zeta`, `rock_mask`, `recorders`, `grid`, `params`) | missing |
| `.../h5` | IID GIFNO runs, `run_*.h5` | 7680 | yes (`run_0.h5`) | `data/gifno_screen/h5` (7680) |
| `.../ood_dipping/h5` | Dipping campaign, plus manifest and outcrop `.npy` files beside `h5/` | 960 | yes | `data/gifno_screen/ood_dipping/h5` (960) |
| `.../ood_three_layer/h5` | Three-layer campaign, same layout | 960 | yes | `data/gifno_screen/ood_three_layer/h5` (960) |
| `.../corner_is/h5` | Corner-instrument suite | 170 | yes | `data/gifno_screen/corner_is/h5` (170) |
| `.../case7_sections/round0/h5` | Round-0 section solves. `round0/` also has `windows/`. | 64 (`sec_0000.h5` …) | yes | `data/vardomain/case7_sections_round0_h5` (64) |
| `.../hallal_vs_2d/` | Hallal ensemble and center-station products, not a raw 2D mesh tree | `ensembles.h5`, `pearson_center.h5`, `tf_2d_center.h5`, `pretell_ensembles.h5` | yes (first three opened) | not checked file-by-file |
| `.../transfer_function/` | Frequency axis, POD vectors, manifests, `models/`, `results/`. No solve H5 directory. | none | n/a | partial under `data/gifno_screen/transfer_function` |

## Not on this mount

No `case7a` directory under the Neural Operator data root. Treasure Island stays local: `data/case7a/opensees_crops/context/h5/full_qvs.h5` and `full_svlab.h5`, with the ten context crops beside them.
