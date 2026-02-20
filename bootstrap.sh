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

# Local (non-NFS) staging directory for pip install.
# pip uses atomic writes (write .tmp → os.replace) which fail on NFS due to
# attribute caching. Installing to /tmp (local SSD) avoids this entirely.
# After a successful install we bulk-copy to the network volume with cp,
# which writes files directly without the atomic-rename pattern.
LOCAL_VENV="/tmp/moss-venv"

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
# Python virtual environment — install to LOCAL disk, then copy to volume.
#
# WHY LOCAL FIRST:
#   pip uses atomic writes: it creates a .tmp file in the same directory as
#   the target and calls os.replace(tmp, target). On NFS, this fails with
#   ENOENT because the directory was just created and NFS attribute caching
#   hasn't propagated it yet. This is not a transient fluke — it will happen
#   consistently for many packages whenever pip creates new dist-info dirs.
#
#   Solution: install to /tmp/moss-venv (local SSD, no NFS). After all
#   packages install successfully, bulk-copy to the network volume with cp.
#   cp writes files directly (no atomic-rename), which works reliably on NFS.
#
# SENTINEL: Written only after the copy succeeds, so any failure leaves
#   no sentinel and the next boot retries from scratch.
# ---------------------------------------------------------------------------
INSTALL_SENTINEL="$VENV_DIR/.install_complete"

if [ ! -f "$INSTALL_SENTINEL" ]; then
    log "=== First-time setup: installing packages to local disk ==="

    # Show available space so we can diagnose if disk is the issue
    log "Available disk space:"
    df -h /tmp "$INSTALL_DIR" 2>/dev/null || true

    # Clean any previous partial local attempt
    log "Creating local venv at $LOCAL_VENV ..."
    rm -rf "$LOCAL_VENV"
    python3.12 -m venv "$LOCAL_VENV"

    # Use the local venv's pip directly — do NOT activate (avoids path issues)
    PIP="$LOCAL_VENV/bin/pip"

    # Install MOSS-TTS package (torch 2.9.1+cu128 and all other deps come
    # from pyproject.toml; the extra-index-url points to the CUDA wheel index).
    log "Installing MOSS-TTS and all dependencies (~10-15 min)..."
    (cd "$SRC_DIR" && "$PIP" install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e .)

    log "Installing flash-attn (prebuilt wheel for torch 2.9, Python 3.12, x86_64)..."
    "$PIP" install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
        || log "WARNING: flash-attn install failed — SDPA fallback will be used."

    log "Installing RunPod serverless runtime..."
    "$PIP" install --no-cache-dir runpod==1.6.1 boto3 botocore || log "WARNING: RunPod runtime install failed."

    log "Installing HuggingFace CLI (hf) and hf_transfer..."
    "$PIP" install --no-cache-dir "huggingface_hub[cli]" hf_transfer || log "WARNING: HuggingFace CLI install failed."

    # ------------------------------------------------------------------
    # All packages installed successfully to local disk.
    # Now bulk-copy to the network volume.
    # cp writes files directly — no atomic rename — so it works on NFS.
    # ------------------------------------------------------------------
    log "Copying venv from local disk to network volume (this may take a few minutes)..."
    log "Available disk space before copy:"
    df -h /tmp "$INSTALL_DIR" 2>/dev/null || true

    mkdir -p "$VENV_DIR"
    log "Starting cp -a $LOCAL_VENV/. $VENV_DIR/ ..."
    cp -av "$LOCAL_VENV/." "$VENV_DIR/" 2>&1 | tail -20
    CP_EXIT=${?}
    log "Copy command exit code: $CP_EXIT"

    if [ $CP_EXIT -ne 0 ]; then
        log "ERROR: Copy failed with exit code $CP_EXIT"
        exit 1
    fi

    log "Copy complete. Verifying..."
    ls -la "$VENV_DIR/bin/python3.12" || { log "ERROR: Python not found in copied venv"; exit 1; }

    log "Cleaning up local temp venv..."
    rm -rf "$LOCAL_VENV"

    # Write sentinel only after copy succeeds
    log "Writing sentinel file: $INSTALL_SENTINEL"
    touch "$INSTALL_SENTINEL"
    ls -la "$INSTALL_SENTINEL" || { log "ERROR: Failed to write sentinel"; exit 1; }
    log "=== First-time setup complete ==="

else
    log "Existing installation found at $VENV_DIR"

    # Patch: ensure runtime deps are up to date on existing volumes.
    # Use 'python -m pip' to avoid shebang path issues in copied venv scripts.
    log "Patching runtime dependencies (no-op if up to date)..."
    "$VENV_DIR/bin/python3.12" -m pip install --no-cache-dir -q \
        runpod==1.6.1 boto3 botocore \
        "huggingface_hub[cli]" hf_transfer || log "WARNING: Patch install failed (non-fatal)."
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
