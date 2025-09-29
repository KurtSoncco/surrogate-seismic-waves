# Surrogate Seismic Waves - AI Coding Guidelines

## Project Overview
This codebase develops machine learning surrogates for seismic wave propagation through layered geological media. We use physics-based simulations (ITASCA FLAC) to generate training data, then train neural networks to predict seismic site responses orders of magnitude faster than traditional finite difference methods.

## Architecture & Data Flow

### Core Components
- **`scripts/PINO/`**: Physics-Informed Neural Operator implementation with PDE-constrained training
- **`scripts/data_generation/`**: Data preprocessing and synthetic dataset creation
- **`scripts/flac/`**: Integration with FLAC physics simulator for ground truth generation
- **`wave_surrogate/models/`**: Model implementations (FNO, DAE, PCE architectures)
- **`wave_surrogate/ttf/`**: Transfer function calculations for seismic response analysis

### Data Pipeline
1. **Input**: Soil profiles (Vs, density) + bedrock motion time series
2. **Physics Simulation**: FLAC generates surface acceleration responses
3. **Feature Extraction**: Transfer functions (TTF) computed via Fourier analysis
4. **ML Training**: Models learn TTF prediction from soil parameters
5. **Validation**: Compare predicted vs. simulated transfer functions

## Critical Workflows

### Environment Setup
```bash
# Use uv for fast, reproducible Python environments
uv venv
uv sync --extra dev  # Includes PyTorch with CUDA support
```

### Training PINO Models
```bash
# From project root
cd scripts/PINO
python pino_main.py
```
- Loads pickled data: `Vs_values_*.pt`, `Rho_values_*.pt`, `TTF_data_*.pt`
- Uses hybrid loss: MSE on transfer functions + PDE residual constraints
- Saves best model to `outputs/models/Soil_Bedrock/`

### Data Generation
- FLAC scripts in `flac_scripts/` configure physics simulations
- Output stored as pickle files in `data/1D Profiles/TF_HLC/`
- Transfer functions computed using `wave_surrogate.ttf.TTF()`

## Project Conventions

### Data Handling
- **Profile Padding**: Use `pad_array()` to standardize soil profile lengths to `config.INPUT_SIZE` (29 layers)
- **NaN Handling**: Replace NaNs with `np.nan_to_num()` before tensor conversion
- **Multi-channel Inputs**: Stack Vs and density as `[2, profile_length]` tensors
- **Normalization**: Spatial coordinates normalized to [0,1], physical scaling handled in PDE loss

### Physics Integration
- **PDE Constraints**: Wave equation `ρu_tt - (G u_z)_z = 0` enforced via `WaveEquationLoss`
- **Material Properties**: Shear modulus `G = ρ * Vs²` computed from Vs/density profiles
- **Boundary Conditions**: Free surface at z=0, fixed bedrock at z=max_depth

### Model Architecture
- **Fourier Neural Operators**: Spectral convolutions for translation-invariant physics
- **Encoder-Decoder**: Convolutional encoders map soil profiles to latent space
- **Hybrid Training**: Data loss on TTFs + physics loss on PDE residuals

### Validation & Metrics
- **Primary Metric**: MSE between predicted and true transfer functions
- **Non-differentiable TTF**: Use `wave_surrogate.ttf.TTF()` for ground truth comparisons
- **Frequency Domain**: Evaluate on log-spaced frequencies from 0.1 to 2.5 Hz

### Logging & Experiment Tracking
- **Weights & Biases**: All training runs logged to `WANDB_PROJECT`
- **Model Checkpoints**: Best validation loss saved automatically
- **Config Tracking**: All hyperparameters logged via `vars(config)`

## Key Files & Patterns

### Configuration
- `scripts/PINO/pino_config.py`: Centralized hyperparameters (learning rates, architectures, data paths)
- Physical constants: `LAYER_THICKNESS = 5.0` meters, `DT = 1e-3` seconds

### Data Loading
```python
# Standard pattern in training scripts
vs_profiles = pickle.load(open(config.VS_PICKLE_PATH, "rb"))
rho_profiles = pickle.load(open(config.RHO_PICKLE_PATH, "rb"))
ttf_data = pickle.load(open(config.TTF_PICKLE_PATH, "rb"))
freq_data = np.loadtxt(config.FREQ_PATH)
```

### Model Instantiation
```python
# PINO with encoder
vs_encoder = Encoder(channels=config.ENCODER_CHANNELS, latent_dim=config.LATENT_DIM)
model = PINO(vs_encoder=vs_encoder, latent_dim=config.LATENT_DIM, ...)
```

### Training Loop
- **Grid Generation**: `get_spatiotemporal_grid()` creates (z,t) coordinates
- **Forward Pass**: `u_pred = model(vs_rho_profiles, input_motion, grid)`
- **Loss Computation**: Data loss on surface/bedrock FFT ratios + PDE constraints
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)`

## Common Pitfalls

### Data Preprocessing
- Always pad profiles to `config.INPUT_SIZE` before stacking channels
- Handle variable-length soil profiles from FLAC output
- Ensure density profiles exist (create dummy if missing)

### Physics Scaling
- PDE loss requires proper normalization: spatial derivatives ÷ L₀, time derivatives ÷ T₀²
- Material interpolation must be differentiable for autograd
- Grid coordinates assumed normalized [0,1] → physical scaling in loss function

### GPU Memory
- Large batches may cause OOM: monitor with `torch.cuda.memory_summary()`
- Spatiotemporal grids: `(BATCH_SIZE, SPATIAL_POINTS=128, TIMESTEPS=1500)`

### Validation
- Training uses differentiable FFT approximations
- Validation uses non-differentiable `TTF()` for accurate comparisons
- Frequency interpolation required to match target frequency bins

## Development Workflow

### Adding New Models
1. Create model class in `wave_surrogate/models/`
2. Add training script in `scripts/`
3. Update config with model-specific hyperparameters
4. Implement physics loss if physics-informed

### Experiment Tracking
- Each run gets unique W&B name via timestamp/config
- Log all metrics: train_data_loss, train_pde_loss, val_mse_numpy
- Save model state dict on validation improvement

### Testing
- Run `uv run pytest` for CI validation
- Currently minimal tests - add physics validation tests
- Integration tests should verify FLAC → ML pipeline end-to-end</content>
<parameter name="filePath">/home/kurt-asus/surrogate-seismic-waves/.github/copilot-instructions.md