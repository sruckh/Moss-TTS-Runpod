# coding=utf-8
# MOSS-TTS RunPod Serverless Configuration

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_SIGNATURE_VERSION = os.environ.get("S3_SIGNATURE_VERSION", "s3v4")
S3_ADDRESSING_STYLE = os.environ.get("S3_ADDRESSING_STYLE", "path")

RUNPOD_VOLUME = "/runpod-volume"
MOSS_DIR = os.environ.get("MOSS_DIR", f"{RUNPOD_VOLUME}/moss-tts")
AUDIO_VOICES_DIR = Path(os.environ.get("AUDIO_VOICES_DIR", f"{MOSS_DIR}/audio_voices"))
OUTPUT_AUDIO_DIR = Path(os.environ.get("OUTPUT_AUDIO_DIR", f"{MOSS_DIR}/output_audio"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", f"{MOSS_DIR}/models"))

DEFAULT_MODEL_REPO = "OpenMOSS-Team/MOSS-TTS"
MODEL_REPO = os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(MODELS_ROOT / MODEL_REPO)))
MODEL_REVISION = os.environ.get("MODEL_REVISION")
DEFAULT_AUDIO_TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
AUDIO_TOKENIZER_REPO = os.environ.get("AUDIO_TOKENIZER_REPO", DEFAULT_AUDIO_TOKENIZER_REPO)
AUDIO_TOKENIZER_DIR = Path(
    os.environ.get("AUDIO_TOKENIZER_DIR", str(MODELS_ROOT / AUDIO_TOKENIZER_REPO))
)
AUDIO_TOKENIZER_REVISION = os.environ.get("AUDIO_TOKENIZER_REVISION")
RUNPOD_HF_CACHE_DIR = Path(
    os.environ.get("RUNPOD_HF_CACHE_DIR", "/runpod-volume/huggingface-cache/hub")
)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
DEFAULT_SAMPLE_RATE = 24000

DEVICE = "cuda" if os.environ.get("DEVICE") != "cpu" else "cpu"
DEFAULT_DTYPE = os.environ.get("DEFAULT_DTYPE", "auto")
DEFAULT_ATTN_IMPLEMENTATION = os.environ.get("DEFAULT_ATTN_IMPLEMENTATION", "auto")
DEFAULT_AUDIO_TOKENIZER_DEVICE = os.environ.get("DEFAULT_AUDIO_TOKENIZER_DEVICE", "cuda")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "4096"))
OOM_TOKEN_CAP_24GB = int(os.environ.get("OOM_TOKEN_CAP_24GB", "4096"))
OOM_RETRY_MAX_NEW_TOKENS = int(os.environ.get("OOM_RETRY_MAX_NEW_TOKENS", "2048"))
MAX_REFERENCE_AUDIO_SECONDS = int(os.environ.get("MAX_REFERENCE_AUDIO_SECONDS", "0"))
MAX_PREFIX_AUDIO_SECONDS = int(os.environ.get("MAX_PREFIX_AUDIO_SECONDS", "0"))

DEFAULT_AUDIO_TEMPERATURE = float(os.environ.get("DEFAULT_AUDIO_TEMPERATURE", "1.7"))
DEFAULT_AUDIO_TOP_P = float(os.environ.get("DEFAULT_AUDIO_TOP_P", "0.8"))
DEFAULT_AUDIO_TOP_K = int(os.environ.get("DEFAULT_AUDIO_TOP_K", "25"))
DEFAULT_AUDIO_REPETITION_PENALTY = float(os.environ.get("DEFAULT_AUDIO_REPETITION_PENALTY", "1.0"))

DEFAULT_ENABLE_CHUNKING = os.environ.get("DEFAULT_ENABLE_CHUNKING", "false").lower() == "true"
DEFAULT_MAX_CHARS_PER_CHUNK = int(os.environ.get("DEFAULT_MAX_CHARS_PER_CHUNK", "300"))
DEFAULT_ENABLE_CROSSFADE = os.environ.get("DEFAULT_ENABLE_CROSSFADE", "true").lower() == "true"
DEFAULT_CROSSFADE_MS = int(os.environ.get("DEFAULT_CROSSFADE_MS", "140"))

# Streaming defaults
DEFAULT_STREAM_MAX_CHARS_PER_CHUNK = int(os.environ.get("DEFAULT_STREAM_MAX_CHARS_PER_CHUNK", "150"))
DEFAULT_STREAM_CROSSFADE_MS = int(os.environ.get("DEFAULT_STREAM_CROSSFADE_MS", "100"))
DEFAULT_CHUNK_PAUSE_MS = int(os.environ.get("DEFAULT_CHUNK_PAUSE_MS", "300"))

CLEANUP_DAYS = int(os.environ.get("CLEANUP_DAYS", "2"))


class Config:
    """Configuration validation and storage."""

    def __init__(self):
        self.validation_errors = []

        self.HF_TOKEN = HF_TOKEN
        self.S3_ENDPOINT_URL = S3_ENDPOINT_URL
        self.S3_ACCESS_KEY_ID = S3_ACCESS_KEY_ID
        self.S3_SECRET_ACCESS_KEY = S3_SECRET_ACCESS_KEY
        self.S3_BUCKET_NAME = S3_BUCKET_NAME
        self.S3_REGION = S3_REGION
        self.S3_SIGNATURE_VERSION = S3_SIGNATURE_VERSION
        self.S3_ADDRESSING_STYLE = S3_ADDRESSING_STYLE

        self.AUDIO_VOICES_DIR = AUDIO_VOICES_DIR
        self.OUTPUT_AUDIO_DIR = OUTPUT_AUDIO_DIR
        self.MODELS_ROOT = MODELS_ROOT
        self.MODEL_REPO = MODEL_REPO
        self.MODEL_DIR = MODEL_DIR
        self.MODEL_REVISION = MODEL_REVISION
        self.AUDIO_TOKENIZER_REPO = AUDIO_TOKENIZER_REPO
        self.AUDIO_TOKENIZER_DIR = AUDIO_TOKENIZER_DIR
        self.AUDIO_TOKENIZER_REVISION = AUDIO_TOKENIZER_REVISION
        self.RUNPOD_HF_CACHE_DIR = RUNPOD_HF_CACHE_DIR

        self.device = DEVICE
        self.default_dtype = DEFAULT_DTYPE
        self.default_attn_implementation = DEFAULT_ATTN_IMPLEMENTATION
        self.default_audio_tokenizer_device = DEFAULT_AUDIO_TOKENIZER_DEVICE
        self.default_max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.oom_token_cap_24gb = OOM_TOKEN_CAP_24GB
        self.oom_retry_max_new_tokens = OOM_RETRY_MAX_NEW_TOKENS
        self.max_reference_audio_seconds = MAX_REFERENCE_AUDIO_SECONDS
        self.max_prefix_audio_seconds = MAX_PREFIX_AUDIO_SECONDS

        self.default_audio_temperature = DEFAULT_AUDIO_TEMPERATURE
        self.default_audio_top_p = DEFAULT_AUDIO_TOP_P
        self.default_audio_top_k = DEFAULT_AUDIO_TOP_K
        self.default_audio_repetition_penalty = DEFAULT_AUDIO_REPETITION_PENALTY

        self.default_enable_chunking = DEFAULT_ENABLE_CHUNKING
        self.default_max_chars_per_chunk = DEFAULT_MAX_CHARS_PER_CHUNK
        self.default_enable_crossfade = DEFAULT_ENABLE_CROSSFADE
        self.default_crossfade_ms = DEFAULT_CROSSFADE_MS

        self.default_stream_max_chars_per_chunk = DEFAULT_STREAM_MAX_CHARS_PER_CHUNK
        self.default_stream_crossfade_ms = DEFAULT_STREAM_CROSSFADE_MS
        self.default_chunk_pause_ms = DEFAULT_CHUNK_PAUSE_MS

        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.AUDIO_EXTS = AUDIO_EXTS

        # Avoid eager CUDA probing during import/startup. On some RunPod hosts
        # this can emit non-actionable warnings before the worker has started.

        try:
            self.AUDIO_VOICES_DIR.mkdir(parents=True, exist_ok=True)
            self.OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            self.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.AUDIO_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.validation_errors.append(f"Failed to create directories: {exc}")

        missing_s3 = [
            name
            for name in ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
            if not getattr(self, name)
        ]
        if missing_s3:
            self.validation_errors.append(f"S3 configuration missing: {', '.join(missing_s3)}")

        log.info("Model repo: %s", self.MODEL_REPO)
        if self.MODEL_REVISION:
            log.info("Model revision: %s", self.MODEL_REVISION)
        log.info("Model dir: %s", self.MODEL_DIR)
        log.info("Audio tokenizer repo: %s", self.AUDIO_TOKENIZER_REPO)
        if self.AUDIO_TOKENIZER_REVISION:
            log.info("Audio tokenizer revision: %s", self.AUDIO_TOKENIZER_REVISION)
        log.info("Audio tokenizer dir: %s", self.AUDIO_TOKENIZER_DIR)
        log.info("RunPod HF cache dir: %s", self.RUNPOD_HF_CACHE_DIR)
        log.info("Device: %s", self.device)

    def validate(self) -> bool:
        if self.validation_errors:
            log.error("Configuration validation failed")
            for err in self.validation_errors:
                log.error("  - %s", err)
            return False
        return True


config = Config()
