# GIFNO-VARDOMAIN

LOGLO-POD on windows 500–1500 m wide, in a **separate experiment** from the
shipped Case 7 models. Checkpoints go to `checkpoints/vardomain/`. Training
refuses to write into `GIFNO-FDO-XT-LOGLO-POD`, `DeepONet-Residual/checkpoints`,
or `tier2_pod64_full7680`.

Two encodings, same data, same POD, same loss:

| encoding | grid | how size is visible |
|---|---|---|
| `reference` | 128 × 500, x/W and z/Lz | FiLM on log W and log H |
| `metric` | 256 × 4000 at 1 m | interior starts at column 1000, edge columns outside it, smoothed mask, FiLM on log W, log H, log f1, log f2 |

## Data (no new OpenSees for width)

Dense labels for the 64 Box sections (every 5 m of surface) live under
`data/vardomain/section_labels/round0/`. They match the existing 500 m window
labels to ~1e-7.

```bash
# rebuild labels (already done locally)
python dense_labels.py \
  --h5-dir "$BOX/case7_sections/round0/h5" \
  --calibration "$BOX/case7_sections/round0/windows/outcrop_calibration.json" \
  --out ../../data/vardomain/section_labels/round0

python build_cache.py \
  --iid-h5-dir "$BOX/h5" --iid-tf-dir "$BOX/transfer_function" \
  --section "round0=$BOX/case7_sections/round0/h5:../../data/vardomain/section_labels/round0" \
  --out ../../data/vardomain/cache
```

## Train (Savio or Lambda)

```bash
# Lambda
bash hpc/lambda_train.sh setup
bash hpc/lambda_train.sh cache
bash hpc/lambda_train.sh launch    # tmux session vardomain

# Savio, one encoding/seed per job
ENCODING=reference SEED=0 sbatch hpc/savio_train.sh
ENCODING=metric    SEED=0 sbatch hpc/savio_train.sh
```

After all six members: `python evaluate.py --summarize checkpoints/vardomain`

## Optional 250 m sections (Stampede)

Only if depth past 150 m matters after the ablation. Writes to
`$SCRATCH/vardomain/deep250`, not to `case7_sections/round0`.

```bash
bash hpc/submit_deep.sh
```

`run_sections.py` reads `depth_m` from the CSV and defaults to 150 m, so the
shipped 150 m generator is unchanged.
