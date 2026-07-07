# GIFNO-FDO-XT-LOGLO-POD

Surrogate for seismic transfer functions: maps a 2D subsurface input
`(B, IN_CHANNELS, NZ_MAX, NX)` to per-recorder spectra `(B, n_recorders, n_freq)`.

The model is a **spectral-bias encoder + POD-DeepONet readout**. After a long
profiling/optimization pass it runs as a **1D-along-x operator** (depth collapsed
to the surface row), which is both faster *and* more accurate than the original
full-depth model.

---

## Architecture

```
input (B, 4, 128, 500)
   │
   ▼  ChannelLift                 1x1 lift  4 -> LATENT_CHANNELS (128)
   │
   ▼  DepthCollapse               (B,C,128,W) -> (B,C,1,W)   einsum weighted sum over depth
   │
   ▼  DualPathLOGLOStack          5 LOGLO layers, now operating on a depth-1 grid
   │     ├─ global path: FFT spectral conv (low-freq, long-range)
   │     ├─ local path:  patch spectral conv (mid-freq, localized)
   │     ├─ HFP:         high-frequency propagation (residual of a blurred copy)
   │     ├─ HF-MLP:      pointwise high-freq mixing
   │     └─ fusion:      combine paths
   │
   ▼  RecorderPODDeepONetHeadXT   project onto precomputed POD basis (POD_NUM_MODES)
   │
   ▼  output (B, n_recorders, n_freq)
```

### Why depth collapse works
The POD readout only consumes the **surface row** of the encoded field, so running
the (cost ~ `H*W`) LOGLO layers at full `NZ_MAX=128` depth is wasted work. The
encoder is collapsed to a single depth row *before* the expensive layers.

`DepthCollapse` is mathematically a depthwise `(NZ_MAX, 1)` convolution, but it is
written as `einsum("bchw,ch->bcw")`. This matters: the large-kernel **depthwise-conv
backward is very slow on GPU**, while the einsum backward is a matmul. Swapping the
conv for the einsum tripled the throughput ceiling on A100 (212 → 634 samples/s).

`DualPathLOGLOStack` and `HighFrequencyPropagation` clamp their depth-axis kernels /
patch sizes to the grid, so the same 2D code degrades gracefully to 1D when depth = 1.

---

## Optimization journey

Starting point: the full-depth model was the training bottleneck on Delta — a 2000-sample
run took ~42 h.

| step | change | what it bought |
|------|--------|----------------|
| 1 | Fine-grained profiling (`record_function` + isolated CUDA micro-benchmarks) | Localized cost to the encoder; `global_fft` dominant; `fp32` FFT unavoidable (`rfftn` has no bf16) |
| 2 | **Early depth downsampling** `NZ_MAX -> NZ_MAX/k` (strided depthwise conv) | 3.4x faster encoder; bigger batches fit; accuracy unchanged |
| 3 | **Full depth collapse to 1D** (`LOGLO_DEPTH_STRIDE = NZ_MAX`) + clamp encoder hparams to depth-1 | 59.5M -> 2.88M params, ~10x less memory; accuracy slightly *better* |
| 4 | **`einsum` `DepthCollapse`** replacing the depthwise conv | removed the conv-backward bottleneck; 3x higher throughput ceiling |

### Final results (A100-SXM4-40GB, `--limit 500`)

| config | params | epoch | best throughput | test_rel_l2 | pearson_rec | logspec | lsd |
|--------|-------:|------:|----------------:|------------:|------------:|--------:|----:|
| original depth-128 | 59.5M | 21.1 s | 18.7 samp/s | 0.405 | 0.909 | 0.335 | 2.272 |
| depth-32 | 59.5M | 5.9 s | 70.3 samp/s | 0.393 | 0.914 | 0.340 | 2.299 |
| depth-1 (depthwise conv) | 2.88M | 2.5 s | 212.6 samp/s | 0.380 | 0.917 | 0.329 | 2.221 |
| **depth-1 einsum, bs=8 (default)** | **2.88M** | **1.4 s** | **634.3 samp/s** | **0.369** | **0.921** | **0.322** | **2.173** |
| depth-1 einsum, bs=64 | 2.88M | 0.8 s | 634.3 samp/s | 0.389 | 0.914 | 0.329 | 2.224 |

Net: **~15x faster per epoch, ~34x higher throughput ceiling, ~20x fewer params, and
better accuracy** than the original. The ~42 h run becomes ~3 h.

After step 4 the remaining cost is the irreducible encoder core: `global_fft` (the
`fp32` FFT, ~52%), `hfp` (~26%), and `fusion` (~24%).

---

## Key config knobs

All overridable via `GIFNO_<KEY>` environment variables (see `config.py`).

| key | default | meaning |
|-----|---------|---------|
| `LOGLO_DEPTH_STRIDE` | `NZ_MAX` (full collapse) | depth reduction `k`: `>=NZ_MAX` → 1D einsum collapse; `>1` → strided depthwise pool; `1` → legacy full depth |
| `LATENT_CHANNELS` | 128 | encoder width |
| `POD_NUM_MODES` | 32 | POD readout modes |
| `LOSS_RADIAL_WEIGHT` | 0.25 | radial spectral loss weight |
| `LOGLO_PATCH_SIZE` | 16,20 | local-patch size (height is clamped to encoder depth = 1 at the default; only width is active) |
| `LOGLO_HF_NOISE_ALPHA` | 0.025 | high-freq noise injection |
| `BATCH_SIZE` | 8 | bs=8 = best accuracy; bs=64 = ~1.7x faster, slight accuracy give-back |

---

## Running

```bash
cd experiments/GIFNO-FDO-XT-LOGLO-POD
source ../GIFNO/delta_env.sh        # Delta env + paths

# single training run (screen, 500 samples)
sbatch delta_train.sh --limit 500

# full-dataset training (no --limit; bump walltime for the full set)
sbatch --time=24:00:00 delta_train.sh

# speed/accuracy profiling
sbatch delta_profile.sh             # or: python profile_speed.py --limit 500 --batches 1,4,16,64
python profile_speed.py --components --limit 500   # per-encoder-block breakdown

# post-hoc eval of a checkpoint
sbatch delta_eval.sh --limit 500

# local dev smoke
bash local_train.sh --limit 500
```

### Sweep

`sweep_variants_loglo_pod.tsv` defines the variants; `sweep_launch.py` submits one
SLURM job each (built on the depth-1 default). Includes a depth ablation
(`depth32`, `depth_full`) to confirm depth-1 still wins at full scale, plus a
`latent256` capacity probe (cheap now that the model is tiny).

```bash
bash delta_sweep.sh --variants sweep_variants_loglo_pod.tsv --limit 2000   # screen
bash delta_sweep.sh --full                                                 # full dataset
bash delta_sweep.sh --dry-run --limit 2000                                 # preview only
bash delta_sweep_rerun.sh loglo_pod_ref --full                             # rerun one variant
```

### Tier-2 stability sweep

`sweep_variants_loglo_pod_stability.tsv` builds on the winning tier-2
convergence band-curriculum config (`loglo_pod_bandcurr_cl`, the best `--limit
2000` run at `test_rel_l2 ~0.301`) and changes one thing per variant to probe
*model stability of the curriculum transition* plus POD/objective capacity:

- **Transition factorial (8 runs):** a full 2^3 over the warm-restart smoothness
  knobs `BAND_CURRICULUM_LR_RESTART_SCALE` x `BAND_CURRICULUM_RESET_OPT_STATE` x
  `BAND_CURRICULUM_RAMP_EPOCHS`. `tier2_base` is the all-default corner (and
  reproduces the prior result); analyze the cube as main effects + interactions.
- **Timing:** `tier2_patience45` (`BAND_CURRICULUM_PHASE_PATIENCE`).
- **POD components:** `tier2_pod48`, `tier2_pod64` (`POD_NUM_MODES`).
- **Objective weight:** `tier2_bbw0.25` (`LOSS_BAND_BALANCED_WEIGHT`).
- **Leading combined candidate:** `tier2_pod48_stable` (POD 48 + gentlest edges).

```bash
bash delta_sweep.sh --variants sweep_variants_loglo_pod_stability.tsv --limit 2000
bash delta_sweep.sh --variants sweep_variants_loglo_pod_stability.tsv --limit 2000 --time 06:00:00
bash delta_sweep.sh --variants sweep_variants_loglo_pod_stability.tsv --limit 2000 --dry-run
bash delta_sweep_rerun.sh tier2_pod48 --variants sweep_variants_loglo_pod_stability.tsv --limit 2000
```

### Capability checks (unseen geometries)

Compare the trained surrogate against OpenSees transfer functions on hand-crafted
seiskit experiments (`three_layer/`, `dipping/`) that probe geometries outside
the training distribution. Ground-truth TFs are cached on disk and skipped on
rerun unless `--force-gt` is passed.

```bash
cd experiments/GIFNO-FDO-XT-LOGLO-POD
export GIFNO_MODEL_DIR=~/surrogate-seismic-waves/checkpoints/tier2_pod64_n2000
export GIFNO_POD_NUM_MODES=64 GIFNO_LATENT_CHANNELS=128 GIFNO_NUM_FNO_LAYERS=5
export GIFNO_DEEPONET_LATENT_DIM=128

# All 5 cases (3 three_layer + 2 dipping) from ~/seiskit/neural-operator/experiments
uv run python capability_check.py --all

# Single case; outputs under ~/surrogate-seismic-waves/checkpoints/capability_checks/
uv run python capability_check.py --h5 ~/seiskit/neural-operator/experiments/three_layer/h5/case_0.h5
```

Requires `seiskit` on `PYTHONPATH` (or `~/seiskit`) for the first ground-truth
TF computation per case. Each case directory contains `tf_true.npy`, `tf_pred.npy`,
`metrics.json`, and comparison plots.

See [`checkpoints/capability_checks/README.md`](../../checkpoints/capability_checks/README.md)
for results interpretation and generalization strategies.
