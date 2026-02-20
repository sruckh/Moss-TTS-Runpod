# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Syntax validation
python3 -m py_compile config.py serverless_engine.py handler.py
bash -n bootstrap.sh

# Docker build
docker build -t moss-tts-runpod .

# Local warmup test (loads model and exits)
python3 handler.py --warmup

# Quick in-process test
python3 -c "
from handler import handler
result = list(handler({'id': 'test', 'input': {'action': 'health_check'}}))
print(result)
"
```

## Confirmed upstream API facts (verified against `clis/moss_tts_app.py`)

- `model.generate()` accepts `audio_temperature`, `audio_top_p`, `audio_top_k`, `audio_repetition_penalty` — these are **correct** custom kwargs for the MOSS-TTS model (not standard Transformers names)
- `processor.build_assistant_message(audio_codes_list=[path])` — correct signature for continuation mode
- `processor.decode(outputs)[0].audio_codes_list[0]` — returns a float waveform tensor (already decoded, ready to save)
- `processor(conversations, mode="generation"|"continuation")` — mode strings are correct
- Flash-attention prebuilt wheel for torch 2.9 + Python 3.12 + x86_64: `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` (added to v2.8.3 release Dec 2025)
- `moss_audio_tokenizer` is a **git submodule** — cloning without `--recurse-submodules` leaves it empty, breaking all imports

## Architecture

Four files form the complete worker:

- **`config.py`** — All defaults and env-var config. The `Config` class validates on import; the module-level `config` singleton is imported everywhere. Add new tunables here first.
- **`serverless_engine.py`** — The `MossTTSInference` class owns model load and generation. A module-level `_inference_engine` singleton is managed by `get_inference_engine()` (lazy, called on first request). Key methods: `generate_audio` (batch, returns numpy array) and `generate_audio_stream_decoded` (streaming, yields base64 PCM chunks).
- **`handler.py`** — RunPod entry point. `handler()` dispatches to `handler_batch()` or `handler_stream()` based on `stream` input field. All input validation, Opus encoding (via ffmpeg subprocess), and S3 upload live here. The module exposes `handler` to `runpod.serverless.start()`.
- **`bootstrap.sh`** — First-boot only: clones upstream MOSS-TTS source, creates a venv at `/runpod-volume/moss-tts/venv`, installs deps, downloads model weights. Subsequent boots skip install steps (guarded by sentinel files) and copy latest worker files from `/opt/moss-tts/` to the volume before starting the handler.

### Separation of concerns rule
- `handler.py`: request parsing, validation, Opus encoding, S3 upload
- `serverless_engine.py`: model loading, conversation building, audio generation
- `config.py`: all defaults and env-based config — never hardcode values elsewhere

### Startup sequence

```
Container starts → bootstrap.sh
  ├─ [first boot] clone MOSS-TTS → create venv → install deps → download model
  └─ [every boot] copy /opt/moss-tts/{handler,config,serverless_engine}.py → volume
                  activate venv → exec python handler.py
```

Model weights are stored at `/runpod-volume/moss-tts/models/` (HF snapshot). Bootstrap log at `/runpod-volume/moss-tts/bootstrap.log`.

### RunPod request envelope

All requests arrive as `{"input": {...}}`. The handler yields response dicts (generator); batch mode yields one dict, streaming mode yields N chunk dicts then a final `{"status": "complete", ...}` dict.

### Audio paths

Reference/prefix audio files are resolved relative to `AUDIO_VOICES_DIR` (default `/runpod-volume/moss-tts/audio_voices`). HTTP/HTTPS URLs are also accepted and downloaded to a temp file. Path traversal (`../`) is blocked. Supported input formats: `.wav .mp3 .m4a .ogg .flac .webm .aac .opus`.

### Small-GPU alternative

For GPUs with <24 GB VRAM, set `MODEL_REPO=OpenMOSS-Team/MOSS-TTS-Local-Transformer`. The `MossTTSDelay-8B` (default) needs 24 GB+.
