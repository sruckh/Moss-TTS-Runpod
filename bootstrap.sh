#!/bin/bash

# MOSS-TTS RunPod bootstrap script.
# Hybrid Environment: Core worker deps in image, heavy ML deps in persistent venv.

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

# Create venv with access to system site-packages (where worker core is pre-installed)
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    log "Creating virtual environment with system site-packages"
    uv venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$INSTALL_DIR/.uv_cache"
mkdir -p "$UV_CACHE_DIR"

# Descriptive environment check
verify_env() {
    log "Verifying dependency availability..."
    python -c "
import sys
try:
    import runpod, boto3, botocore
    print(f'Core worker deps OK: runpod={runpod.__version__}, boto3={boto3.__version__}')
    import torch, transformers
    print(f'ML deps OK: torch={torch.__version__}, transformers={transformers.__version__}')
except ImportError as e:
    print(f'CRITICAL: {e}', file=sys.stderr)
    sys.exit(1)
"
}

SETUP_MARKER="$VENV_DIR/.setup_complete"
if [ ! -f "$SETUP_MARKER" ] || ! verify_env &>/dev/null; then
    log "Environment missing or corrupted; synchronizing dependencies..."
    check_resources

    # --refresh forces uv to verify all files exist on disk, fixing network volume corruption
    if ! (cd "$SRC_DIR" && uv pip install \
        --refresh \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        -e .); then
        log "ERROR: Installation failed"
        exit 1
    fi

    if [ "$ENABLE_FLASH_ATTN" = "true" ]; then
        log "Attempting optional flash-attn install"
        uv pip install flash-attn || log "flash-attn install failed; continuing without it"
    fi

    verify_env
    sync
    touch "$SETUP_MARKER"
else
    log "Virtual environment verified"
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

log "Starting RunPod handler"
exec python "$SRC_DIR/handler.py"
