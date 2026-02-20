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

# Note: Removed complex /tmp staging approach - installing directly to volume.

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
# Python virtual environment — simple install like upstream.
# Just 3 steps: clone, install, optionally flash-attn.
# ---------------------------------------------------------------------------
INSTALL_SENTINEL="$VENV_DIR/.install_complete"

if [ ! -f "$INSTALL_SENTINEL" ]; then
    log "=== First-time setup: creating venv and installing packages ==="

    # Show available space
    log "Available disk space:"
    df -h "$INSTALL_DIR" 2>/dev/null || true

    # Step 1: Create venv directly on volume
    log "Creating venv at $VENV_DIR ..."
    rm -rf "$VENV_DIR"
    python3.12 -m venv "$VENV_DIR"

    # Step 2: Install MOSS-TTS and all dependencies
    log "Installing MOSS-TTS and all dependencies (~10-15 min)..."
    (cd "$SRC_DIR" && "$VENV_DIR/bin/pip" install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e .)

    # Step 3: Optionally install flash-attn (non-fatal)
    log "Installing flash-attn (optional, will use SDPA fallback if this fails)..."
    "$VENV_DIR/bin/pip" install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
        || log "WARNING: flash-attn install failed — SDPA fallback will be used."

    # Install RunPod runtime
    log "Installing RunPod serverless runtime..."
    "$VENV_DIR/bin/pip" install --no-cache-dir runpod==1.6.1 boto3 botocore || log "WARNING: RunPod runtime install failed."

    # Install HuggingFace CLI
    log "Installing HuggingFace CLI and hf_transfer..."
    "$VENV_DIR/bin/pip" install --no-cache-dir huggingface_hub hf_transfer || log "WARNING: HuggingFace CLI install failed."

    # Write sentinel
    touch "$INSTALL_SENTINEL"
    log "=== First-time setup complete ==="

else
    log "Existing installation found at $VENV_DIR"
fi

# ---------------------------------------------------------------------------
# Download model weights (first boot or after MODEL_REPO change)
# ---------------------------------------------------------------------------
if [ ! -f "$MODEL_DIR/config.json" ]; then
    log "Downloading model '$MODEL_REPO' to $MODEL_DIR ..."
    export HF_HUB_ENABLE_HF_TRANSFER=1

    # Use snapshot_download API directly — avoids shebang path issues with hf CLI.
    "$VENV_DIR/bin/python3.12" - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download
model_repo = os.environ.get("MODEL_REPO", "$MODEL_REPO")
model_dir  = "$MODEL_DIR"
token      = os.environ.get("HF_TOKEN") or None
print(f"[hf] Downloading {model_repo} → {model_dir}", flush=True)
try:
    snapshot_download(model_repo, local_dir=model_dir, token=token)
    print("[hf] Download complete.", flush=True)
except Exception as e:
    print(f"[hf] Download failed: {e}", file=sys.stderr, flush=True)
    # NON-FATAL: let bootstrap continue so we can at least start the handler
    print("[hf] Model download is non-fatal — will retry on next boot.", flush=True)
PYEOF
    log "Model download step complete (may have failed — non-fatal)."
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
# Run the venv's Python directly (avoids activate script shebang path issues).
# ---------------------------------------------------------------------------
log "Starting RunPod handler..."
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
exec "$VENV_DIR/bin/python3.12" "$SRC_DIR/handler.py"
