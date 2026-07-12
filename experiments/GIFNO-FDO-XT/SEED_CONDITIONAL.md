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

# After training, gate on seed robustness (sample_id=0 and 1–2 more points):
uv run python seed_robustness/seed_robustness_check.py --sample-id 0 --max-seeds 30
```

Targets vs current `xt_lat128_d128`: hold rel_l2_central ≲ 0.30; lift
σ_ln(pred)/σ_ln(truth) ≳ 0.7; hold peak |sur−OO| median.

Legacy `xt_lat128_d128` eval: leave `GIFNO_SEED_CONDITIONAL_RECIPE` unset
(SCALE_SPLIT_VS=false, single-path FNO, surface/none activation).
