# Latent FNO Experiment

This experiment tests the hypothesis that operating Fourier Neural Operators (FNO) in a learned latent space can be more effective than operating directly on raw data.

## Architecture

The pipeline consists of three modular components:

1. **Encoder**: Maps input data (e.g., Vs profiles) to a latent representation
2. **FNO Processor**: Applies Fourier Neural Operator transformations in the latent space
3. **Decoder**: Maps the processed latent representation back to output data (e.g., transfer functions)

```
Input (Vs profile) → Encoder → Latent Space → FNO Processor → Latent Space → Decoder → Output (Transfer Function)
```

## Key Hypothesis

Operating FNO in the latent space may be beneficial because:
- The latent space captures the most relevant features for the task
- FNO can learn complex relationships in a lower-dimensional, more structured space
- The latent representation may be more amenable to frequency-domain operations

## Modular Design

Each component is modular and can be easily swapped:

### Encoders
- `MLPEncoder`: Multi-layer perceptron
- `CNNEncoder`: Convolutional neural network
- `TransformerEncoder`: Transformer-based encoder
- `AutoEncoderEncoder`: Pre-trained autoencoder

### FNO Processors
- `SimpleFNOProcessor`: Basic FNO on latent vector
- `SequenceFNOProcessor`: FNO on latent as sequence
- `MultiScaleFNOProcessor`: Multi-scale FNO processing
- `AdaptiveFNOProcessor`: Attention-based adaptive FNO
- `ConditionalFNOProcessor`: Condition-aware FNO processing

### Decoders
- `MLPDecoder`: Multi-layer perceptron
- `CNNDecoder`: Convolutional decoder
- `TransformerDecoder`: Transformer-based decoder
- `AutoEncoderDecoder`: Pre-trained autoencoder
- `FNOOperatorDecoder`: FNO-based decoder

## Usage

### Basic Usage

```python
from latent_FNO.pipeline import create_pipeline

# Create a basic pipeline
model = create_pipeline(
    input_dim=29,      # Vs profile length
    output_dim=1000,   # Transfer function length
    latent_dim=128,    # Latent space dimension
    encoder_type="mlp",
    decoder_type="mlp",
    fno_processor_type="simple"
)

# Forward pass
output = model(input_data)
```

### Using Predefined Configurations

```python
from latent_FNO.config import get_config

# Get a predefined configuration
config = get_config("cnn_encoder")

# Create model from config
model = create_pipeline(
    input_dim=config.input_dim,
    output_dim=config.output_dim,
    latent_dim=config.latent_dim,
    encoder_type=config.encoder_type,
    decoder_type=config.decoder_type,
    fno_processor_type=config.fno_processor_type,
    encoder_config=config.encoder_config,
    decoder_config=config.decoder_config,
    fno_processor_config=config.fno_processor_config
)
```

### Training

```python
from latent_FNO.train import LatentFNOTrainer
from latent_FNO.config import LatentFNOConfig

# Create configuration
config = LatentFNOConfig()

# Create trainer
trainer = LatentFNOTrainer(config)

# Load data
train_loader, val_loader, test_loader = trainer.load_data()

# Train model
results = trainer.train(train_loader, val_loader)

# Evaluate
test_metrics = trainer.evaluate(test_loader)
```

## Experiment Configurations

### Available Configurations

- `baseline`: Basic MLP encoder/decoder with simple FNO
- `cnn_encoder`: CNN encoder with MLP decoder
- `transformer_encoder`: Transformer encoder with MLP decoder
- `fno_decoder`: MLP encoder with FNO decoder
- `sequence_fno`: Sequence-based FNO processor
- `multiscale_fno`: Multi-scale FNO processor
- `adaptive_fno`: Adaptive FNO processor
- `high_latent_dim`: Higher latent dimension (256)
- `low_latent_dim`: Lower latent dimension (64)

### Custom Configuration

```python
from latent_FNO.config import create_custom_config

config = create_custom_config(
    latent_dim=256,
    encoder_type="transformer",
    decoder_type="fno_operator",
    fno_processor_type="adaptive",
    learning_rate=5e-4
)
```

## Ablation Studies

The pipeline supports easy ablation studies:

```python
from latent_FNO.pipeline import AblationStudyPipeline

# Full pipeline
full_model = AblationStudyPipeline(
    input_dim=29, output_dim=1000, latent_dim=128,
    use_encoder=True, use_fno=True, use_decoder=True
)

# No FNO (encoder + decoder only)
no_fno_model = AblationStudyPipeline(
    input_dim=29, output_dim=1000, latent_dim=128,
    use_encoder=True, use_fno=False, use_decoder=True
)

# Encoder only
encoder_only = AblationStudyPipeline(
    input_dim=29, output_dim=1000, latent_dim=128,
    use_encoder=True, use_fno=False, use_decoder=False
)
```

## File Structure

```
latent_FNO/
├── __init__.py
├── main.py            # Main script for running experiments
├── test_new_structure.py  # Test script for new structure
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py         # Encoder implementations
│   │   ├── decoders.py         # Decoder implementations
│   │   ├── fno_processor.py    # FNO processor implementations
│   │   ├── pipeline.py         # Main pipeline class
│   │   └── train.py           # Training script and trainer class
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── wandb_utils.py      # Weights & Biases utilities
│   │   ├── metrics.py          # Comprehensive metrics and evaluation
│   │   └── data_utils.py       # Data loading and preprocessing
│   └── configs/
│       ├── __init__.py
│       └── config.py           # Configuration classes and predefined configs
├── tests/
│   └── test_comprehensive.py   # Comprehensive test suite
├── models/            # Saved model checkpoints
├── results/           # Experiment results and plots
└── README.md          # This file
```

## Key Features

- **Modular Design**: Easy to swap components and test different architectures
- **Configurable**: Extensive configuration options for all components
- **Ablation Support**: Built-in support for ablation studies
- **Training Infrastructure**: Complete training pipeline with validation, checkpointing, and logging
- **Multiple FNO Variants**: Different approaches to applying FNO in latent space
- **Pre-trained Components**: Support for using pre-trained encoders/decoders
- **Comprehensive Metrics**: Extensive evaluation metrics including correlation analysis, frequency-dependent metrics, and statistical tests
- **W&B Integration**: Full Weights & Biases integration for experiment tracking and visualization
- **Data Utilities**: Advanced data loading, preprocessing, and augmentation utilities
- **Command-line Interface**: Easy-to-use CLI for running different experiments

## Running Experiments

### Command Line Interface

The main script provides a comprehensive CLI for running experiments:

```bash
# List available configurations
python main.py list-configs

# Quick functionality test
python main.py test

# Train a model with specific configuration
python main.py train --config baseline --wandb --epochs 1000

# Run ablation study
python main.py ablation

# Compare different configurations
python main.py compare

# Train with custom parameters
python main.py train --config sequence_fno --batch-size 64 --epochs 500
```

### Individual Components

You can also use individual components directly:

```python
# Test new structure
python test_new_structure.py

# Run comprehensive tests
python tests/test_comprehensive.py

# Train with specific configuration
from src.models.train import LatentFNOTrainer
from src.configs.config import get_config

config = get_config("baseline")
trainer = LatentFNOTrainer(config)
# ... training code
```

### Available Configurations

- `baseline`: Basic MLP encoder/decoder with simple FNO
- `cnn_encoder`: CNN encoder with MLP decoder
- `transformer_encoder`: Transformer encoder with MLP decoder
- `fno_decoder`: MLP encoder with FNO decoder
- `sequence_fno`: Sequence-based FNO processor
- `multiscale_fno`: Multi-scale FNO processor
- `adaptive_fno`: Adaptive FNO processor
- `high_latent_dim`: Higher latent dimension (256)
- `low_latent_dim`: Lower latent dimension (64)

## Expected Benefits

This latent FNO approach should provide:
- Better generalization by operating in a learned feature space
- More efficient learning through lower-dimensional representations
- Improved interpretability of the learned transformations
- Potential for transfer learning across different domains

The modular design allows systematic testing of these hypotheses across different encoder/decoder/FNO combinations.
