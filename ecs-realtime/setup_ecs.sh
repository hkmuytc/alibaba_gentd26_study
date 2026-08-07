#!/bin/bash
set -e

echo "========================================="
echo " ECS Serverless AI Experiment Setup"
echo "========================================="

# 1. Update package list and install system dependencies
echo "[1/4] Installing system dependencies (tmux, python3-pip, etc.)..."
# Suppress interactive prompts like "Which services should be restarted?"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -yq tmux python3-pip python3-venv htop jq

# 2. Check NVIDIA Drivers
echo "[2/4] Verifying NVIDIA Driver & CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found! Please ensure you chose an 'AI/GPU Accelerated' Ubuntu image."
    exit 1
fi
nvidia-smi | grep "CUDA Version" || echo "Warning: Could not detect CUDA version from nvidia-smi."

# 3. Python Virtual Environment & Packages
echo "[3/4] Creating Python Virtual Environment and installing libraries..."
python3 -m venv ecs_env
source ecs_env/bin/activate

# Install PyTorch with CUDA explicitly (standard for Ubuntu)
pip install --upgrade pip
pip install torch torchvision torchaudio
# Install pipeline and data processing libraries
pip install diffusers transformers accelerate pandas numpy scikit-learn "huggingface_hub[cli]" peft xformers

# 4. Final Verification
echo "[4/4] Verifying PyTorch GPU access..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ PyTorch sees GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ ERROR: PyTorch cannot see the GPU!')
    exit(1)
"

echo "========================================="
echo "✅ Setup Complete!"
echo "To start the experiment, run: bash run_experiment.sh"
echo "========================================="
