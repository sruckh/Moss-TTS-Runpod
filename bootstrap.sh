#!/bin/bash

# MOSS-TTS RunPod bootstrap script.
# Optimized for reliability and space efficiency on small network volumes.

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

    log "Installing all dependencies (from pyproject.toml + extras)..."
    # We rely on pyproject.toml for torch and other core deps.
    # We use --no-cache to avoid filling up the ephemeral disk or the network volume.
    # We provide the index-url so the specific +cu128 versions can be resolved.
    (cd "$SRC_DIR" && uv pip install --no-cache \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        -e . \
        runpod==1.6.1 \
        boto3 botocore hf_transfer)

    log "=== Virtual environment setup complete ==="
else
    log "Activating existing virtual environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Quick integrity check
    if ! python -c "import torch, boto3, botocore, runpod" &>/dev/null; then
        log "WARNING: Environment appears broken. Attempting repair..."
        # Repair using uv --no-cache for efficiency
        (cd "$SRC_DIR" && uv pip install --no-cache \
            --index-url https://download.pytorch.org/whl/cu128 \
            --extra-index-url https://pypi.org/simple \
            --index-strategy unsafe-best-match \
            -e . \
            runpod==1.6.1 \
            boto3 botocore hf_transfer) || true
    fi
fi

# Export environment variables
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
