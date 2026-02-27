#!/bin/bash
# MOSS-TTS RunPod Serverless Bootstrap
#
# First boot (~10-20 min, longer if model pre-download is enabled):
#   - Clones MOSS-TTS source + initialises moss_audio_tokenizer submodule
#   - Installs all Python packages via uv in copy mode (NFS-safe)
#   - Optionally pre-downloads model weights from HuggingFace
#
# Subsequent boots (~seconds):
#   - Copies latest handler files from Docker image
#   - Skips install (sentinel guards against partial installs)
#   - Starts the RunPod handler

set -Eeuo pipefail
trap 'echo "[$(date '\''+%Y-%m-%d %H:%M:%S'\'')] ERROR at line $LINENO: \"$BASH_COMMAND\" exited with status $?" >&2' ERR

echo "=== MOSS-TTS RunPod Bootstrap Starting ==="

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INSTALL_DIR="${INSTALL_DIR:-/runpod-volume/moss-tts}"
DOCKER_SRC="${DOCKER_SRC:-/opt/moss-tts}"
SRC_DIR="$INSTALL_DIR/src"
VENV_DIR="$INSTALL_DIR/venv"
SENTINEL="$VENV_DIR/.install_complete"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-}"
MODEL_REPO="${MODEL_REPO:-OpenMOSS-Team/MOSS-TTS}"
MODEL_DIR="${MODEL_DIR:-}"
MOSS_REPO="${MOSS_REPO:-https://github.com/OpenMOSS/MOSS-TTS.git}"
MOSS_REF="${MOSS_REF:-main}"
BOOTSTRAP_DOWNLOAD_MODEL="${BOOTSTRAP_DOWNLOAD_MODEL:-false}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if ! mkdir -p "$INSTALL_DIR" 2>/dev/null || [ ! -w "$INSTALL_DIR" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: INSTALL_DIR '$INSTALL_DIR' is not writable. Falling back to /tmp/moss-tts"
    INSTALL_DIR="/tmp/moss-tts"
    mkdir -p "$INSTALL_DIR"
fi

SRC_DIR="$INSTALL_DIR/src"
VENV_DIR="$INSTALL_DIR/venv"
SENTINEL="$VENV_DIR/.install_complete"
AUDIO_VOICES_DIR="${AUDIO_VOICES_DIR:-$INSTALL_DIR/audio_voices}"
OUTPUT_AUDIO_DIR="${OUTPUT_AUDIO_DIR:-$INSTALL_DIR/output_audio}"
MODEL_DIR="${MODEL_DIR:-$INSTALL_DIR/models/$MODEL_REPO}"

LOG_FILE="$INSTALL_DIR/bootstrap.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Install dir: $INSTALL_DIR"
log "Source dir:  $SRC_DIR"
log "Venv dir:    $VENV_DIR"
log "Sentinel:    $SENTINEL"
log "Model repo:  $MODEL_REPO"
log "Bootstrap download model: $BOOTSTRAP_DOWNLOAD_MODEL"
log "Docker source: $DOCKER_SRC"

for required_file in handler.py config.py serverless_engine.py; do
    if [ ! -f "$DOCKER_SRC/$required_file" ]; then
        log "ERROR: required file not found: $DOCKER_SRC/$required_file"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------------------------
mkdir -p "$AUDIO_VOICES_DIR" "$OUTPUT_AUDIO_DIR" "$MODEL_DIR"

# ---------------------------------------------------------------------------
# Clone MOSS-TTS source (first boot only)
# ---------------------------------------------------------------------------
if [ ! -d "$SRC_DIR/.git" ]; then
    log "Cloning MOSS-TTS (ref: $MOSS_REF)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$MOSS_REF" "$MOSS_REPO" "$SRC_DIR"

    log "Initialising git submodule (moss_audio_tokenizer)..."
    git -C "$SRC_DIR" submodule update --init --recursive
    log "Clone complete."
else
    log "MOSS-TTS source already present at $SRC_DIR"
fi

# ---------------------------------------------------------------------------
# Always copy latest handler files from Docker image so code updates
# take effect on every boot without needing to wipe the network volume.
# ---------------------------------------------------------------------------
log "Copying handler files from Docker image..."
cp "$DOCKER_SRC/handler.py" "$DOCKER_SRC/config.py" "$DOCKER_SRC/serverless_engine.py" "$SRC_DIR/"

# ---------------------------------------------------------------------------
# Python virtual environment
#
# SENTINEL: $VENV_DIR/.install_complete
# Written only after ALL package installs succeed. Using bin/activate as
# sentinel is wrong: it is created at venv init time, before any packages
# are installed. A failed mid-install leaves bin/activate present, so the
# next boot skips install and the handler crashes in a restart loop.
#
# INSTALL TOOL: uv with UV_LINK_MODE=copy
# In copy mode uv writes files directly to site-packages without the
# temp-file-rename atomic write that pip uses. This avoids the NFS
# attribute-cache race that causes pip to fail on RunPod network volumes.
# UV_CACHE_DIR=/tmp keeps the download cache on local SSD.
#
# MOSS-TTS: installed with --no-deps to skip gradio and the full UI
# dependency tree. Only the inference deps from pyproject.toml are
# installed explicitly below.
# ---------------------------------------------------------------------------
if [ ! -f "$SENTINEL" ]; then
    log "=== First-time setup: installing Python environment ==="
    df -h "$INSTALL_DIR" || true

    # Create venv only if it does not already exist.
    # Do NOT try to clear/wipe a partial venv: rm -rf and venv --clear
    # both fail on NFS with "Directory not empty". Instead, reuse the
    # existing venv — uv is idempotent and skips already-installed
    # packages, so a retry picks up exactly where it left off.
    if [ ! -f "$VENV_DIR/bin/python3.12" ]; then
        log "Creating virtual environment..."
        python3.12 -m venv "$VENV_DIR"
    else
        log "Resuming install on existing partial venv..."
    fi
    source "$VENV_DIR/bin/activate"

    # uv is pre-installed in the Docker image at the system level.
    # Keep download cache on local disk; use copy mode for NFS safety.
    export UV_LINK_MODE=copy
    export UV_CACHE_DIR="/tmp/uv-cache"

    log "Installing PyTorch + torchaudio (CUDA 12.8)..."
    uv pip install \
        torch==2.9.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

    log "Installing flash-attn prebuilt wheel (non-fatal)..."
    uv pip install \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
        || log "WARNING: flash-attn failed — SDPA fallback will be used."

    log "Installing MOSS-TTS source package (--no-deps skips gradio / UI packages)..."
    uv pip install --no-deps -e "$SRC_DIR"

    log "Installing torchcodec (PyTorch index only)..."
    uv pip install "torchcodec==0.8.1" \
        --index-url https://download.pytorch.org/whl/cu128

    log "Installing MOSS-TTS inference dependencies (PyPI)..."
    uv pip install \
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
        psutil packaging

    log "Installing RunPod serverless runtime..."
    uv pip install runpod==1.6.1 boto3 botocore huggingface_hub hf_transfer

    # Sentinel written ONLY after every install above succeeds.
    # If anything above exits non-zero, set -e kills bootstrap before we
    # reach this line, the sentinel is never written, and the next boot
    # retries installation on the existing partial venv.
    touch "$SENTINEL"
    log "=== Python environment install complete ==="
else
    log "Virtual environment ready (sentinel found). Skipping install."
fi

# ---------------------------------------------------------------------------
# Download model weights (first boot only)
# ---------------------------------------------------------------------------
download_model_on_boot="$(echo "$BOOTSTRAP_DOWNLOAD_MODEL" | tr '[:upper:]' '[:lower:]')"
if [ "$download_model_on_boot" = "true" ] || [ "$download_model_on_boot" = "1" ]; then
    if [ ! -f "$MODEL_DIR/config.json" ]; then
        log "Downloading model weights ($MODEL_REPO)..."
        HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV_DIR/bin/python3.12" - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL_DIR",
    token=os.environ.get("HF_TOKEN") or None,
)
print("Model download complete.")
PYEOF
    else
        log "Model already present at $MODEL_DIR"
    fi
else
    log "Skipping model download on bootstrap (BOOTSTRAP_DOWNLOAD_MODEL=$BOOTSTRAP_DOWNLOAD_MODEL)"
fi

if [ -f "$MODEL_DIR/config.json" ]; then
    log "✓ config.json found — model looks intact."
elif [ "$download_model_on_boot" = "true" ] || [ "$download_model_on_boot" = "1" ]; then
    log "WARNING: config.json not found at $MODEL_DIR"
    ls -la "$MODEL_DIR" || true
else
    log "Model files not pre-downloaded; first inference request will download from HuggingFace."
fi

# ---------------------------------------------------------------------------
# Start the RunPod handler
# ---------------------------------------------------------------------------
log "Starting RunPod handler..."
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
exec "$VENV_DIR/bin/python3.12" "$SRC_DIR/handler.py"
