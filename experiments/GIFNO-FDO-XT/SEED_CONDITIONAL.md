# Seed-conditional XT recipe

Enable the full plan (depth branch, softplus |TF|, Vs macro/RF channels,
dual-path FNO, seed-contrast + σ_ln aux) with:

```bash
export GIFNO_SEED_CONDITIONAL_RECIPE=1
export GIFNO_LATENT_CHANNELS=128 GIFNO_DEEPONET_LATENT_DIM=128
export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/xt_seed_conditional
export GIFNO_RESULTS_DIR=~/surrogate-seismic-waves/checkpoints/xt_seed_conditional/results

cd experiments/GIFNO-FDO-XT
uv run python main.py   # or sweep row `xt_seed_conditional`
```

## Loss space (important)

**Primary rel-L2 / H1 use linear |TF|**, same as baseline `xt_lat128_d128` and
LOGLO eval (`LOG_TF_LOSS=false`). Do not enable `GIFNO_LOG_TF_LOSS=true` for
this recipe: training in log-amplitude compresses dynamic range, yields overly
smooth / sinusoidal-looking spectra, and hurts Pearson correlation.

Seed-contrast aux also uses **linear Δ|TF|** between replicates. Only the
σ_ln calibration term operates in log-space (by definition).

## After training

```bash
uv run python seed_robustness/seed_robustness_check.py --sample-id 0 --max-seeds 30
```

Targets vs `xt_lat128_d128`: hold rel_l2_central ≲ 0.30; lift
σ_ln(pred)/σ_ln(truth) ≳ 0.7; hold peak |sur−OO| median.

Legacy `xt_lat128_d128` eval: leave `GIFNO_SEED_CONDITIONAL_RECIPE` unset.
