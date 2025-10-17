#!/bin/bash
"""
Setup script for running latent FNO experiments on supercomputers/clusters.
This script ensures all dependencies are properly installed and configured.
"""

echo "🚀 Setting up Latent FNO Experiment Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the experiments/latent_FNO directory"
    exit 1
fi

# Check Python version
echo "🔍 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 not found!"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA detected - installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "  No CUDA detected - installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install wandb numpy matplotlib scipy pandas scikit-learn tqdm

# Check wandb login
echo "🔐 Checking wandb authentication..."
wandb whoami
if [ $? -ne 0 ]; then
    echo "⚠️  Wandb not logged in. Please run: wandb login"
    echo "   You can get your API key from: https://wandb.ai/authorize"
fi

# Make scripts executable
chmod +x run_all_experiments.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run all experiments:"
echo "  source .venv/bin/activate"
echo "  python run_all_experiments.py"
echo ""
echo "To run a single experiment:"
echo "  source .venv/bin/activate"
echo "  python main.py train --config baseline --wandb"
echo ""
echo "Available configurations:"
python main.py list-configs
