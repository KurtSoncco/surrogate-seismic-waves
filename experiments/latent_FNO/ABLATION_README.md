# CNN Encoder Ablation Study

This directory contains scripts for running comprehensive ablation studies on the CNN encoder latent FNO model, which achieved the best performance with a mean sample correlation of **0.64912**.

## 🎯 Goal
Optimize the CNN encoder hyperparameters to achieve even better performance than the baseline correlation of 0.64912.

## 📁 Available Scripts

### 1. `run_quick_validation.py` ⚡
**Purpose**: Quick validation of key hyperparameters
**Duration**: ~2-3 hours
**Experiments**: 5 key configurations
**Use when**: You want to quickly test if the approach works

```bash
python run_quick_validation.py
```

### 2. `run_focused_ablation.py` 🎯
**Purpose**: Focused ablation study on most promising hyperparameters
**Duration**: ~8-12 hours  
**Experiments**: ~50 targeted configurations
**Use when**: You want a comprehensive but manageable study

```bash
python run_focused_ablation.py
```

### 3. `run_ablation_study.py` 🔬
**Purpose**: Complete exhaustive ablation study
**Duration**: ~24-48 hours
**Experiments**: ~200+ configurations
**Use when**: You want to explore all possible combinations

```bash
python run_ablation_study.py
```

### 4. `run_all_experiments.py` 🚀
**Purpose**: Run all original model configurations
**Duration**: ~6-8 hours
**Experiments**: 9 different model types
**Use when**: You want to compare different architectures

```bash
python run_all_experiments.py
```

## 🔧 Setup for Supercomputer

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- wandb account and API key

### Quick Setup
```bash
# Make scripts executable
chmod +x *.py

# Run setup script (if available)
./setup_supercomputer.sh

# Or manual setup
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wandb numpy matplotlib scipy pandas scikit-learn tqdm

# Login to wandb
wandb login
```

## 📊 Hyperparameters Being Tested

### 1. Latent Dimension
- **Range**: 64, 96, 128, 160, 192, 224, 256, 320, 384, 512
- **Impact**: High - affects model capacity and FNO processing

### 2. FNO Processor Parameters
- **Modes**: 8, 12, 16, 20, 24, 28, 32
- **Width**: 32, 48, 64, 80, 96, 112, 128
- **Layers**: 2, 3, 4, 5, 6
- **Impact**: High - core of the FNO processing

### 3. CNN Architecture
- **Channels**: [1,16,32,64], [1,32,64,128], [1,64,128,256], [1,32,64,128,256]
- **Kernels**: [3,3,3], [5,5,5], [3,5,7], [7,5,3]
- **Pooling**: [2,2,2], [2,2,4], [4,2,2], [2,4,2]
- **Impact**: Medium - affects feature extraction

### 4. Training Parameters
- **Learning Rate**: 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2
- **Batch Size**: 32, 64, 128, 256, 512
- **Weight Decay**: 0.0, 1e-5, 1e-4, 1e-3, 1e-2
- **Dropout**: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
- **Impact**: Medium - affects training dynamics

## 📈 Expected Results

### Quick Validation (5 experiments)
- **Time**: 2-3 hours
- **Goal**: Confirm approach works
- **Expected**: Find 1-2 promising configurations

### Focused Ablation (50 experiments)  
- **Time**: 8-12 hours
- **Goal**: Find optimal hyperparameters
- **Expected**: Achieve correlation > 0.70

### Complete Ablation (200+ experiments)
- **Time**: 24-48 hours
- **Goal**: Exhaustive search
- **Expected**: Find best possible configuration

## 🎯 Success Metrics

- **Primary**: Mean sample correlation > 0.64912
- **Secondary**: Test MSE < 14.5, Test MAE < 1.4
- **Tertiary**: Training stability and convergence

## 📁 Output Files

- `validation_results.json` - Quick validation results
- `focused_ablation_results.json` - Focused ablation results  
- `ablation_results.json` - Complete ablation results
- `wandb_logs/` - Detailed experiment logs

## 🌐 Monitoring

All experiments are logged to wandb:
- **Project**: `latent_fno`
- **Dashboard**: https://wandb.ai/kurtwal98-university-of-california-berkeley/latent_fno

## 🚀 Recommended Workflow

1. **Start with quick validation** to confirm the approach
2. **Run focused ablation** if validation shows promise
3. **Run complete ablation** if you need exhaustive search
4. **Analyze results** and select best configuration
5. **Run final validation** with best hyperparameters

## 💡 Tips for Supercomputer

- Use `screen` or `tmux` for long-running experiments
- Monitor GPU memory usage
- Set appropriate time limits for job schedulers
- Use multiple GPUs if available (modify batch size accordingly)
- Check wandb logs regularly for progress

## 🔍 Troubleshooting

- **CUDA out of memory**: Reduce batch size
- **Training too slow**: Increase batch size or reduce model size
- **Poor convergence**: Adjust learning rate or add regularization
- **Wandb errors**: Check internet connection and API key
