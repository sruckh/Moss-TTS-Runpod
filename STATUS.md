# MOSS-TTS RunPod Serverless - Implementation Status

## Project Overview
- **Repository**: https://github.com/sruckh/Moss-TTS-Runpod
- **Model**: OpenMOSS-Team/MOSS-TTS (MossTTSDelay-8B)
- **Platform**: RunPod Serverless GPU infrastructure

## Completed Work

### 1. Initial Setup (Commit: 3f5eadd)
- Created project structure with Dockerfile, bootstrap.sh, config.py, handler.py, serverless_engine.py
- Implemented lazy model loading with CUDA support
- Implemented batch mode TTS with OGG/Opus encoding and S3 upload
- Implemented voice cloning and continuation mode
- Added health check endpoint
- Created comprehensive README with architecture diagrams

### 2. Streaming Mode (Commit: 5890a59)
- Added streaming mode support with `handler_stream()` function
- Implemented `generate_audio_stream_decoded()` in serverless_engine.py
- Streaming yields base64-encoded int16 PCM chunks incrementally
- Added crossfade tail buffering for smooth chunk transitions
- Added chunk_pause_ms for natural pacing between chunks
- Updated README with streaming documentation and examples

## File Structure
```
/opt/docker/Moss-TTS/
├── Dockerfile              # RunPod base:1.0.3-cuda1281-ubuntu2404
├── bootstrap.sh            # First-boot setup (clone MOSS-TTS, install deps, download model)
├── config.py               # Environment configuration and validation
├── handler.py              # RunPod serverless handler (batch + streaming modes)
├── serverless_engine.py    # MOSS-TTS inference wrapper
├── requirements.txt        # Runtime deps (runpod, boto3, huggingface-hub)
├── README.md               # Documentation with diagrams
├── IMPLEMENTATION.md       # Implementation guide from IndexTTS2 reference
├── TODO.md                 # Task checklist (all marked complete)
├── STATUS.md               # This file
└── docs/diagrams/
    ├── architecture.drawio # System architecture diagram
    ├── architecture.svg
    ├── data-flow.drawio
    └── data-flow.svg
```

## API Reference

### Supported MOSS-TTS Parameters
- `text` (required) - Text to synthesize (max 10,000 chars)
- `mode` - `generation` or `continuation` (default: generation)
- `reference_audio` - Path or URL to reference audio for voice cloning
- `prefix_audio` - Required when mode=continuation
- `expected_tokens` - Expected duration in tokens (1s ≈ 12.5 tokens)
- `max_new_tokens` - Maximum tokens to generate (128-8192, default: 4096)
- `audio_temperature` - Sampling temperature (default: 1.7 for Delay, 1.0 for Local)
- `audio_top_p` - Nucleus sampling (default: 0.8 for Delay, 0.95 for Local)
- `audio_top_k` - Top-k sampling (default: 25 for Delay, 50 for Local)
- `audio_repetition_penalty` - Repetition penalty (default: 1.0)

### Handler-Level Features
- `enable_chunking` - Split long text into chunks (default: false)
- `max_chars_per_chunk` - Characters per chunk (50-1000, default: 300)
- `enable_crossfade` - Crossfade between chunks (default: true)
- `crossfade_ms` - Crossfade duration (default: 140)

### Streaming Parameters
- `stream` - Enable streaming mode (default: false)
- `output_format` - Only `pcm_16` supported
- `stream_max_chars_per_chunk` - Streaming chunk size (default: 150)
- `stream_crossfade_ms` - Streaming crossfade (default: 100)
- `chunk_pause_ms` - Silence between chunks (default: 300)

## Environment Variables

### Required
- `S3_ENDPOINT_URL`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`

### Recommended
- `HF_TOKEN`
- `MODEL_REPO` (default: OpenMOSS-Team/MOSS-TTS)
- `AUDIO_VOICES_DIR` (default: /runpod-volume/moss-tts/audio_voices)
- `OUTPUT_AUDIO_DIR` (default: /runpod-volume/moss-tts/output_audio)

## Deployment Status
- ✅ Code compiles without errors
- ✅ Container builds successfully
- ✅ Pushed to GitHub (public repo)
- ❌ **NOT YET TESTED on RunPod Serverless**

## Next Steps for Testing
1. Deploy container to RunPod Serverless
2. Attach network volume
3. Set environment variables
4. Test health check endpoint: `{"input": {"action": "health_check"}}`
5. Test batch mode generation
6. Test voice cloning with reference audio
7. Test continuation mode
8. Test streaming mode
9. Verify S3 upload and presigned URLs

## Important Notes
- MOSS-TTS uses different defaults than IndexTTS2 (see audio_temperature, audio_top_p, audio_top_k)
- Model loads on first request (cold start ~2-5 minutes)
- GPU with 24GB+ VRAM recommended for MossTTSDelay-8B
- For smaller GPUs, switch to OpenMOSS-Team/MOSS-TTS-Local-Transformer
