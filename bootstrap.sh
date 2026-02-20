#!/bin/bash

# MOSS-TTS RunPod Serverless Bootstrap Script
#
# First boot (~20-30 min):
#   - clones MOSS-TTS source + initialises git submodule (moss_audio_tokenizer)
#   - installs all Python packages to LOCAL disk (/tmp) to avoid NFS failures
#   - copies completed venv from /tmp to the network volume (one-time bulk copy)
#   - downloads model weights from HuggingFace
#
# Subsequent boots (~seconds):
#   - copies latest handler files from Docker image
#   - starts the RunPod handler using the persisted venv

set -e  # Exit on any error

echo "=== MOSS-TTS RunPod Bootstrap Starting ==="

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/moss-tts}"
DOCKER_SRC="/opt/moss-tts"
SRC_DIR="$INSTALL_DIR/src"
VENV_DIR="$INSTALL_DIR/venv"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"
MODEL_REPO="${MODEL_REPO:-OpenMOSS-Team/MOSS-TTS}"
MODEL_DIR="${MODEL_DIR:-$INSTALL_DIR/models/$MODEL_REPO}"
MOSS_REPO="${MOSS_REPO:-https://github.com/OpenMOSS/MOSS-TTS.git}"
MOSS_REF="${MOSS_REF:-main}"

# ---------------------------------------------------------------------------
# Logging: tee every line to log file AND stdout so RunPod console + file both
# show the same output.
# ---------------------------------------------------------------------------
LOG_FILE="$INSTALL_DIR/bootstrap.log"
mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Log file: $LOG_FILE"
log "Install dir: $INSTALL_DIR"
log "Source dir:  $SRC_DIR"
log "Venv dir:    $VENV_DIR"
log "Model repo:  $MODEL_REPO"
log "Model dir:   $MODEL_DIR"
log "MOSS ref:    $MOSS_REF"

# ---------------------------------------------------------------------------
# Ensure required directories exist early
# ---------------------------------------------------------------------------
log "Creating required directories..."
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$MODEL_DIR"

# ---------------------------------------------------------------------------
# Clone MOSS-TTS source (first boot only)
# The repo contains a git submodule (moss_audio_tokenizer) that MUST be
# initialised — without it the audio tokenizer module is empty and all
# imports fail.
# ---------------------------------------------------------------------------
if [ ! -d "$SRC_DIR/.git" ]; then
    log "Cloning MOSS-TTS source (ref: $MOSS_REF)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$MOSS_REF" "$MOSS_REPO" "$SRC_DIR"

    log "Initialising git submodule (moss_audio_tokenizer)..."
    git -C "$SRC_DIR" submodule update --init --recursive
    log "Source cloned and submodules initialised."
else
    log "MOSS-TTS source already present at $SRC_DIR"
fi

# ---------------------------------------------------------------------------
# Always copy latest handler files from Docker image so that code updates
# take effect on every boot without needing to rebuild the network volume.
# ---------------------------------------------------------------------------
log "Copying handler files from Docker image..."
cp "$DOCKER_SRC/handler.py"           "$SRC_DIR/"
cp "$DOCKER_SRC/config.py"            "$SRC_DIR/"
cp "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

# ---------------------------------------------------------------------------
# Python virtual environment — following IndexTTS2 proven pattern
# ---------------------------------------------------------------------------
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log "=== First-time setup: creating virtual environment ==="

    # Show available space
    df -h "$INSTALL_DIR" 2>/dev/null || true

    log "Creating venv at $VENV_DIR ..."
    rm -rf "$VENV_DIR"
    python3.12 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    log "Installing uv package manager..."
    pip install --no-cache-dir uv
    export UV_LINK_MODE=copy

    log "Installing PyTorch (CUDA 12.8)..."
    uv pip install torch==2.9.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 || \
        pip install --no-cache-dir torch==2.9.1 torchaudio==2.9.1 \
            --index-url https://download.pytorch.org/whl/cu128

    log "Installing flash-attn (prebuilt wheel)..."
    pip install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
        || log "WARNING: flash-attn failed — SDPA fallback will be used."

    log "Installing MOSS-TTS..."
    (cd "$SRC_DIR" && pip install --no-cache-dir -e .)

    log "Installing RunPod serverless runtime..."
    pip install --no-cache-dir runpod==1.6.1 boto3 botocore

    log "Installing HuggingFace CLI and hf_transfer..."
    pip install --no-cache-dir huggingface_hub hf_transfer

    log "=== Virtual environment setup complete ==="
else
    log "Activating existing virtual environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# ---------------------------------------------------------------------------
# Download model weights (first boot or after MODEL_REPO change)
# ---------------------------------------------------------------------------
if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "Model not found. Downloading from HuggingFace..."

    if [ -n "$HF_TOKEN" ]; then
        hf download "$MODEL_REPO" --local-dir="$MODEL_DIR" --token="$HF_TOKEN" || {
            log "WARNING: Failed to download with token, trying without..."
            hf download "$MODEL_REPO" --local-dir="$MODEL_DIR"
        }
    else
        log "HF_TOKEN not set, downloading without authentication..."
        hf download "$MODEL_REPO" --local-dir="$MODEL_DIR"
    fi

    log "Model download complete."
else
    log "Model already present at $MODEL_DIR"
fi

# Verify model
if [ -f "$MODEL_DIR/config.json" ]; then
    log "✓ config.json found — model looks intact."
else
    log "WARNING: config.json not found at $MODEL_DIR after download."
    ls -la "$MODEL_DIR" || true
fi

# ---------------------------------------------------------------------------
# Start the RunPod handler
# ---------------------------------------------------------------------------
log "Starting RunPod handler..."
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
exec python "$SRC_DIR/handler.py"
