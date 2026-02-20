#!/bin/bash

# MOSS-TTS RunPod bootstrap script.
# Scaffolded after working indexTTS2 implementation for maximum reliability on network volumes.

set -e  # Exit on any error

echo "=== MOSS-TTS RunPod Bootstrap Starting ==="

# Configuration
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/moss-tts}"
DOCKER_SRC="/opt/moss-tts"
SRC_DIR="${SRC_DIR:-$INSTALL_DIR/src}"
VENV_DIR="${VENV_DIR:-$INSTALL_DIR/venv}"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"
MODEL_REPO="${MODEL_REPO:-OpenMOSS-Team/MOSS-TTS}"
MODEL_DIR="${MODEL_DIR:-$INSTALL_DIR/models/$MODEL_REPO}"
MODELS_ROOT="${MODELS_ROOT:-$INSTALL_DIR/models}"

# Git clone settings
MOSS_REPO="${MOSS_REPO:-https://github.com/OpenMOSS/MOSS-TTS.git}"
MOSS_REF="${MOSS_REF:-main}"

# Logging
LOG_FILE="$INSTALL_DIR/bootstrap.log"
mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Log file: $LOG_FILE"
log "Install directory: $INSTALL_DIR"
log "Source directory: $SRC_DIR"
log "Venv directory: $VENV_DIR"

# Ensure required directories exist
log "Ensuring required directories exist..."
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$MODELS_ROOT"

# Clone source
if [ ! -d "$SRC_DIR/.git" ]; then
    log "Cloning MOSS-TTS source into $SRC_DIR..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$MOSS_REF" "$MOSS_REPO" "$SRC_DIR"
else
    log "Source already present at $SRC_DIR"
fi

# Always copy latest worker files from Docker image
log "Updating worker files from Docker image..."
cp "$DOCKER_SRC/handler.py" "$SRC_DIR/"
cp "$DOCKER_SRC/config.py" "$SRC_DIR/"
cp "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

# Create Python virtual environment and install dependencies (first time only)
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log "=== First-time setup: creating virtual environment ==="
    python3.12 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    log "Installing uv package manager..."
    pip install --no-cache-dir uv
    export UV_LINK_MODE=copy

    log "Installing PyTorch (CUDA 12.8)..."
    uv pip install torch==2.9.1+cu128 torchaudio==2.9.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 || \
        pip install --no-cache-dir torch==2.9.1+cu128 torchaudio==2.9.1+cu128 \
            --index-url https://download.pytorch.org/whl/cu128

    log "Installing MOSS-TTS from source (--no-deps)..."
    (cd "$SRC_DIR" && pip install --no-cache-dir --no-deps -e .)

    log "Installing runtime dependencies..."
    pip install --no-cache-dir \
        "transformers==5.0.0" \
        "safetensors==0.6.2" \
        "numpy==2.1.0" \
        "orjson==3.11.4" \
        "tqdm==4.67.1" \
        "PyYAML==6.0.3" \
        "einops==0.8.1" \
        "scipy==1.16.2" \
        "librosa==0.11.0" \
        "tiktoken==0.12.0" \
        "psutil" "packaging" "ninja" "setuptools" "wheel" "gradio"

    log "Installing RunPod and serverless dependencies..."
    pip install --no-cache-dir \
        runpod==1.6.1 \
        boto3 botocore hf_transfer

    log "=== Virtual environment setup complete ==="
else
    log "Activating existing virtual environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Patch: ensure critical deps are actually functional
    if ! python -c "import torch, boto3, botocore, runpod" &>/dev/null; then
        log "WARNING: Environment check failed. Attempting quick repair..."
        pip install --no-cache-dir boto3 botocore runpod hf_transfer || true
    fi
fi

# Make sure the source is importable
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check if model exists, if not download it
if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "Model not found. Downloading $MODEL_REPO..."
    mkdir -p "$MODEL_DIR"

    if [ -n "${HF_TOKEN:-}" ]; then
        hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --token "$HF_TOKEN" || {
            log "WARNING: Download with token failed, trying without..."
            hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
        }
    else
        log "HF_TOKEN not set, downloading without auth..."
        hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    fi
    log "Model download complete"
else
    log "Model already present at $MODEL_DIR"
fi

# Start handler
log "Starting RunPod handler..."
exec python "$SRC_DIR/handler.py"
