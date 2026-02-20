#!/bin/bash

# MOSS-TTS RunPod bootstrap script.
# Following proven installation sequence for reliability and space efficiency.

set -euo pipefail

echo "=== MOSS-TTS RunPod Bootstrap Starting ==="

INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/moss-tts}"
SRC_DIR="$INSTALL_DIR/src"
VENV_DIR="$INSTALL_DIR/venv"

# 1. Ensure project root exists
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 2. Setup temporary directory on volume to avoid boot drive limits
export TMPDIR="$INSTALL_DIR/.tmp"
mkdir -p "$TMPDIR"

# 3. Clone or update source
if [ ! -d "src" ]; then
    echo "Cloning MOSS-TTS source..."
    git clone https://github.com/OpenMOSS/MOSS-TTS.git src
fi

# 4. Sync worker files from Docker image
echo "Updating worker files from image..."
cp /opt/moss-tts/*.py src/

# 5. Virtual Environment Management
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 6. User-Prescribed Installation Sequence
# These steps follow the project's own instructions for CUDA 12.8
cd src

echo "Step 1: Installing MOSS-TTS core..."
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -e .

echo "Step 2: Installing MOSS-TTS flash-attn..."
# This step handles the optional flash-attn requirement correctly
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]"

# 7. Worker and S3 Runtime Layer
echo "Step 3: Installing RunPod and S3 dependencies..."
pip install --no-cache-dir runpod==1.6.1 boto3 botocore huggingface-hub hf_transfer

# 8. Model Weights Management
MODEL_REPO="${MODEL_REPO:-OpenMOSS-Team/MOSS-TTS}"
MODEL_DIR="$INSTALL_DIR/models/$MODEL_REPO"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading model $MODEL_REPO..."
    mkdir -p "$MODEL_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    if [ -n "${HF_TOKEN:-}" ]; then
        huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR" --token "$HF_TOKEN" || \
        huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    else
        huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    fi
fi

# 9. Clean up temp space
rm -rf "$TMPDIR"/* || true

# 10. Start Handler
echo "Starting RunPod handler..."
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
exec python handler.py
