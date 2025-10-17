#!/usr/bin/env python3
"""
Script to run all latent FNO experiments sequentially.
This ensures proper resource management and avoids conflicts.

This script is portable and can run on:
- Local machines
- Supercomputers/clusters
- Any system with Python and the required dependencies

Requirements:
- Python 3.8+
- PyTorch with CUDA support (if GPUs available)
- wandb (for experiment tracking)
- All dependencies from requirements.txt
"""

import subprocess
import sys
import time
import os
import platform

def run_experiment(config_name: str) -> bool:
    """Run a single experiment and return success status."""
    print(f"\n🚀 Starting experiment: {config_name}")
    print("-" * 50)
    
    try:
        # Run the training with wandb enabled, activating virtual environment
        result = subprocess.run([
            'bash', '-c', 
            'source /home/kurt-asus/surrogate-seismic-waves/.venv/bin/activate && python main.py train --config ' + config_name + ' --wandb'
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ {config_name} completed successfully!")
            return True
        else:
            print(f"❌ {config_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {config_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"💥 {config_name} failed with exception: {e}")
        return False

def check_environment():
    """Check the environment and print system information."""
    print('🔍 Environment Check')
    print('-' * 30)
    print(f'Python version: {sys.version}')
    print(f'Platform: {platform.platform()}')
    print(f'Working directory: {os.getcwd()}')
    
    # Check for CUDA/GPU availability
    try:
        import torch
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA version: {torch.version.cuda}')
            print(f'GPU count: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        else:
            print('⚠️  No CUDA GPUs detected - will use CPU')
    except ImportError:
        print('❌ PyTorch not installed!')
        return False
    
    # Check for wandb
    try:
        import wandb
        print(f'Wandb version: {wandb.__version__}')
    except ImportError:
        print('❌ Wandb not installed!')
        return False
    
    print('✅ Environment check passed!')
    return True

def main():
    """Run all experiments sequentially."""
    # Check environment first
    if not check_environment():
        print('❌ Environment check failed. Please install required dependencies.')
        return False
    
    # List of all configurations to run
    configs = [
        'baseline',
        'cnn_encoder', 
        'transformer_encoder',
        'fno_decoder',
        'sequence_fno',
        'multiscale_fno',
        'adaptive_fno',
        'high_latent_dim',
        'low_latent_dim'
    ]
    
    print('\n🎯 Latent FNO Comprehensive Experiment Suite')
    print('=' * 60)
    print(f'Running {len(configs)} configurations sequentially...')
    print('Each experiment will be logged to wandb project: latent_fno')
    print('=' * 60)
    
    results = {}
    successful = 0
    failed = 0
    
    for i, config in enumerate(configs, 1):
        print(f'\n[{i}/{len(configs)}] Processing: {config}')
        
        # Run the experiment
        success = run_experiment(config)
        results[config] = success
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Brief pause between experiments
        if i < len(configs):
            print("\n⏳ Waiting 10 seconds before next experiment...")
            time.sleep(10)
    
    # Print final summary
    print('\n' + '=' * 60)
    print('🎉 EXPERIMENT SUITE COMPLETED!')
    print('=' * 60)
    print(f'✅ Successful: {successful}/{len(configs)}')
    print(f'❌ Failed: {failed}/{len(configs)}')
    
    print('\n📊 Results Summary:')
    for config, success in results.items():
        status = "✅" if success else "❌"
        print(f'  {status} {config}')
    
    print('\n🌐 View results at: https://wandb.ai/kurtwal98-university-of-california-berkeley/latent_fno')
    
    return successful == len(configs)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
