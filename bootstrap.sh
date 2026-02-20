#!/bin/bash

# MOSS-TTS RunPod Serverless Bootstrap Script
#
# First boot (~10-20 min):
#   - clones MOSS-TTS source + initialises git submodule (moss_audio_tokenizer)
#   - creates Python venv and installs all dependencies
#   - downloads model weights from HuggingFace
#
# Subsequent boots (~seconds):
#   - activates existing venv
#   - copies latest handler files from Docker image
#   - starts the RunPod handler

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
cp "$DOCKER_SRC/handler.py"          "$SRC_DIR/"
cp "$DOCKER_SRC/config.py"           "$SRC_DIR/"
cp "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

# ---------------------------------------------------------------------------
# Python virtual environment (first boot only)
#
# SENTINEL: We use a dedicated marker file written only AFTER all pip installs
# succeed. Using $VENV_DIR/bin/activate as the guard is unsafe because it is
# created at venv-creation time, before any packages are installed. A failed
# pip install would leave activate present but packages missing, causing
# RunPod to restart into a broken venv → infinite reboot loop.
# ---------------------------------------------------------------------------
INSTALL_SENTINEL="$VENV_DIR/.install_complete"

if [ ! -f "$INSTALL_SENTINEL" ]; then
    log "=== First-time setup: creating virtual environment ==="

    # Remove any partial venv left by a previous failed attempt so we start clean.
    if [ -d "$VENV_DIR" ]; then
        log "Removing incomplete venv from previous attempt..."
        rm -rf "$VENV_DIR"
    fi

    python3.12 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # Install MOSS-TTS package (torch 2.9.1+cu128 and all other deps come
    # from pyproject.toml; the extra-index-url points to the CUDA wheel index).
    # Note: pyproject.toml also pulls in `gradio` as a base dependency; this
    # is upstream behaviour and cannot be avoided without patching the source.
    log "Installing MOSS-TTS package and dependencies (this takes ~10-15 min)..."
    (cd "$SRC_DIR" && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e .)

    log "Installing flash-attn (prebuilt wheel for torch 2.9, Python 3.12, x86_64)..."
    pip install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

    log "Installing RunPod serverless runtime..."
    pip install --no-cache-dir \
        runpod==1.6.1 \
        boto3 \
        botocore

    log "Installing HuggingFace CLI and hf_transfer..."
    pip install --no-cache-dir "huggingface-hub[cli]" hf_transfer

    # All installs succeeded — write sentinel so subsequent boots skip this block.
    touch "$INSTALL_SENTINEL"
    log "=== Virtual environment setup complete ==="

else
    log "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate"

    # Patch: install any newly added runtime dependencies on existing volumes.
    # These are fast no-ops if the packages are already present.
    # flash-attn is included here so it is retried if a previous boot failed
    # to install it (e.g. network error) — || true keeps this non-fatal.
    log "Patching runtime dependencies (no-op if up to date)..."
    pip install --no-cache-dir -q \
        runpod==1.6.1 boto3 botocore \
        "huggingface-hub[cli]" hf_transfer \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
        2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Download model weights (first boot or after MODEL_REPO change)
# ---------------------------------------------------------------------------
if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "Downloading model '$MODEL_REPO' to $MODEL_DIR ..."
    export HF_HUB_ENABLE_HF_TRANSFER=1

    if [ -n "${HF_TOKEN:-}" ]; then
        huggingface-cli download "$MODEL_REPO" \
            --local-dir "$MODEL_DIR" \
            --token "$HF_TOKEN" || {
            log "Token download failed; retrying without token..."
            huggingface-cli download "$MODEL_REPO" \
                --local-dir "$MODEL_DIR"
        }
    else
        log "HF_TOKEN not set; downloading without authentication..."
        huggingface-cli download "$MODEL_REPO" \
            --local-dir "$MODEL_DIR"
    fi
    log "Model download complete."
else
    log "Model already present at $MODEL_DIR"
fi

# Sanity-check
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
