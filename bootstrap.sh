#!/bin/bash

# MOSS-TTS RunPod bootstrap script.
# First boot installs source, Python environment, and model weights onto /runpod-volume.

set -euo pipefail

echo "=== MOSS-TTS RunPod Bootstrap Starting ==="

INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/moss-tts}"
DOCKER_SRC="/opt/moss-tts"
SRC_DIR="${SRC_DIR:-$INSTALL_DIR/src}"
VENV_DIR="${VENV_DIR:-$INSTALL_DIR/venv}"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"
MODEL_REPO="${MODEL_REPO:-OpenMOSS-Team/MOSS-TTS}"
MODEL_DIR="${MODEL_DIR:-$INSTALL_DIR/models/$MODEL_REPO}"
MODELS_ROOT="${MODELS_ROOT:-$INSTALL_DIR/models}"

MOSS_REPO="${MOSS_REPO:-https://github.com/OpenMOSS/MOSS-TTS.git}"
MOSS_REF="${MOSS_REF:-main}"
ENABLE_FLASH_ATTN="${ENABLE_FLASH_ATTN:-false}"

LOG_FILE="$INSTALL_DIR/bootstrap.log"
mkdir -p "$INSTALL_DIR"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Resource diagnostics
check_resources() {
    log "--- Resource Status ---"
    free -h || true
    df -h / /runpod-volume || true
    log "-----------------------"
}

log "Log file: $LOG_FILE"
log "Install directory: $INSTALL_DIR"
log "Source directory: $SRC_DIR"
log "Venv directory: $VENV_DIR"
log "Model repo: $MODEL_REPO"
log "Model directory: $MODEL_DIR"

check_resources

log "Ensuring required directories exist"
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$MODELS_ROOT"

if [ ! -d "$SRC_DIR/.git" ]; then
    log "Cloning MOSS-TTS source into $SRC_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$MOSS_REF" "$MOSS_REPO" "$SRC_DIR"
else
    log "Source already present at $SRC_DIR"
fi

log "Copying worker files from Docker image"
cp "$DOCKER_SRC/handler.py" "$SRC_DIR/"
cp "$DOCKER_SRC/config.py" "$SRC_DIR/"
cp "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log "Creating virtual environment"
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$INSTALL_DIR/.uv_cache"
mkdir -p "$UV_CACHE_DIR"

# Check if environment is actually functional
is_env_broken() {
    log "Verifying core dependencies..."
    if ! python -c "import boto3; import botocore; import runpod; import torch; import transformers; print('Core imports successful')" 2>&1; then
        return 0 # Broken
    fi
    return 1 # OK
}

SETUP_MARKER="$VENV_DIR/.setup_complete"
SHOULD_INSTALL=0

if [ ! -f "$SETUP_MARKER" ]; then
    log "Setup marker missing; triggering fresh install"
    SHOULD_INSTALL=1
elif is_env_broken; then
    log "Virtual environment is corrupted or incomplete; wiping and recreating"
    rm -rf "$VENV_DIR"
    uv venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    SHOULD_INSTALL=1
fi

if [ "$SHOULD_INSTALL" -eq 1 ]; then
    log "Installing MOSS-TTS dependencies with uv"
    check_resources

    log "Installing MOSS-TTS and runtime extras"
    # We use a single command to ensure atomic dependency resolution
    if ! (cd "$SRC_DIR" && uv pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        -e . \
        runpod==1.6.1 \
        boto3 \
        botocore \
        huggingface-hub \
        hf_transfer); then
        log "ERROR: Installation failed"
        exit 1
    fi

    log "Verifying environment integrity after install"
    uv pip check || log "WARNING: uv pip check reported issues"
    
    if is_env_broken; then
        log "ERROR: Environment still broken after installation"
        exit 1
    fi

    if [ "$ENABLE_FLASH_ATTN" = "true" ]; then
        log "Attempting optional flash-attn install"
        uv pip install flash-attn || log "flash-attn install failed; continuing without it"
    fi

    check_resources
    sync
    touch "$SETUP_MARKER"
else
    log "Virtual environment verified; proceeding to model check"
fi

export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "Model snapshot not found. Downloading $MODEL_REPO to $MODEL_DIR"
    mkdir -p "$MODEL_DIR"

    if [ -n "${HF_TOKEN:-}" ]; then
        hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --token "$HF_TOKEN" || {
            log "Download with token failed. Retrying without token"
            hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
        }
    else
        log "HF_TOKEN not set; downloading without auth"
        hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    fi
else
    log "Model already present at $MODEL_DIR"
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "WARNING: config.json not found in $MODEL_DIR"
    ls -la "$MODEL_DIR" || true
fi

log "Starting RunPod handler"
exec python "$SRC_DIR/handler.py"
