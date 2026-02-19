# MOSS-TTS RunPod Serverless

> Scalable Text-to-Speech inference server powered by [OpenMOSS-Team/MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) on RunPod Serverless GPU infrastructure.

A production-ready serverless worker that provides high-quality neural TTS with voice cloning capabilities, automatic S3 uploads, and efficient resource utilization through lazy model loading.

## Features

- **Neural TTS Generation** - State-of-the-art speech synthesis using MOSS-TTS transformer models
- **Voice Cloning** - Clone any voice using reference audio files
- **Continuation Mode** - Extend existing audio while maintaining voice consistency
- **Advanced Decoding Controls** - Fine-tune output with temperature, top-p, top-k, and repetition penalty
- **Long-Form Support** - Optional text chunking with crossfade for extended content
- **OGG/Opus Encoding** - Efficient audio compression via FFmpeg
- **S3 Integration** - Automatic uploads to S3-compatible storage with presigned URLs
- **Lazy Model Loading** - Model loads on first request, minimizing cold-start time
- **Health Check Endpoint** - Monitor system status, GPU availability, and configuration
- **Network Volume Persistence** - Model weights and dependencies cached on RunPod network volumes

## Architecture

![Architecture Diagram](./docs/diagrams/architecture.svg)

The system consists of four main components:

1. **handler.py** - RunPod serverless entry point, request validation, S3 upload
2. **serverless_engine.py** - Inference engine with lazy model loading and MOSS-TTS integration
3. **config.py** - Environment-based configuration with validation
4. **bootstrap.sh** - First-boot setup script for dependencies and model weights

## Data Flow

![Data Flow Diagram](./docs/diagrams/data-flow.svg)

1. Client sends request with text and optional reference audio
2. Handler validates parameters (text length, mode, audio files, decoding ranges)
3. Model loads lazily on first request (cached for subsequent requests)
4. Conversation built based on mode (generation or continuation)
5. Model generates audio tokens with configured decoding parameters
6. Audio decoded from tokens and optionally chunked/crossfaded
7. Audio encoded to OGG/Opus via FFmpeg
8. File uploaded to S3, presigned URL generated
9. Response returned with URL and metadata

## Quick Start

### Prerequisites

- RunPod account with Serverless enabled
- S3-compatible storage (or any S3-compatible service)
- GPU with 24GB+ VRAM recommended (for `MossTTSDelay-8B`)

### Building the Container

```bash
# Clone the repository
git clone https://github.com/your-org/Moss-TTS.git
cd Moss-TTS

# Build and push to container registry
docker build -t your-registry/moss-tts:latest .
docker push your-registry/moss-tts:latest
```

### Deploying to RunPod

1. Create a new **Serverless Endpoint** in RunPod
2. Use the built container image
3. Attach a **Network Volume** (for model caching)
4. Configure environment variables (see below)

### Environment Variables

#### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `S3_ENDPOINT_URL` | S3-compatible API endpoint | `https://s3.amazonaws.com` |
| `S3_ACCESS_KEY_ID` | S3 access key | `AKIAIOSFODNN7EXAMPLE` |
| `S3_SECRET_ACCESS_KEY` | S3 secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `S3_BUCKET_NAME` | Target bucket name | `my-tts-output` |

#### Recommended

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace authentication token | - |
| `MODEL_REPO` | HuggingFace model repository | `OpenMOSS-Team/MOSS-TTS` |
| `MOSS_REF` | Git branch/commit for MOSS-TTS source | `main` |
| `AUDIO_VOICES_DIR` | Directory for reference audio files | `/runpod-volume/moss-tts/audio_voices` |
| `OUTPUT_AUDIO_DIR` | Directory for temporary output | `/runpod-volume/moss-tts/output_audio` |
| `MODEL_DIR` | Model weights location | `/runpod-volume/moss-tts/models/OpenMOSS-Team/MOSS-TTS` |

#### Optional Decoding Defaults

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_AUDIO_TEMPERATURE` | Sampling temperature (0-5) | `1.7` |
| `DEFAULT_AUDIO_TOP_P` | Nucleus sampling threshold (0-1) | `0.8` |
| `DEFAULT_AUDIO_TOP_K` | Top-k sampling (1-200) | `25` |
| `DEFAULT_AUDIO_REPETITION_PENALTY` | Repetition penalty (0.8-2.0) | `1.0` |
| `DEFAULT_MAX_NEW_TOKENS` | Maximum tokens to generate (128-8192) | `4096` |
| `DEFAULT_ENABLE_CHUNKING` | Enable text chunking for long content | `false` |
| `DEFAULT_MAX_CHARS_PER_CHUNK` | Characters per chunk when chunking enabled | `300` |
| `DEFAULT_ENABLE_CROSSFADE` | Enable crossfade between chunks | `true` |
| `DEFAULT_CROSSFADE_MS` | Crossfade duration in milliseconds | `140` |
| `CLEANUP_DAYS` | Days before auto-deleting temp files | `2` |

## Usage

### Direct Generation

```json
{
  "input": {
    "text": "Hello from MOSS-TTS! This is a test of the text-to-speech system.",
    "mode": "generation"
  }
}
```

### Voice Cloning

```json
{
  "input": {
    "text": "This should sound like the reference speaker.",
    "mode": "generation",
    "reference_audio": "speaker_demo.wav"
  }
}
```

Place reference audio files in the configured `AUDIO_VOICES_DIR` or provide HTTP/HTTPS URLs.

### Continuation Mode

```json
{
  "input": {
    "text": "<original transcript> And here is the continuation of the speech.",
    "mode": "continuation",
    "prefix_audio": "previous_segment.wav",
    "reference_audio": "speaker_demo.wav"
  }
}
```

### With Custom Decoding Parameters

```json
{
  "input": {
    "text": "Testing custom decoding settings.",
    "mode": "generation",
    "reference_audio": "speaker.wav",
    "max_new_tokens": 2048,
    "audio_temperature": 1.5,
    "audio_top_p": 0.85,
    "audio_top_k": 30,
    "audio_repetition_penalty": 1.1
  }
}
```

### Long-Form Content with Chunking

```json
{
  "input": {
    "text": "This is a very long text that would benefit from chunking...",
    "mode": "generation",
    "reference_audio": "speaker.wav",
    "enable_chunking": true,
    "max_chars_per_chunk": 300,
    "enable_crossfade": true,
    "crossfade_ms": 140
  }
}
```

### Health Check

```json
{
  "input": {
    "action": "health_check"
  }
}
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "checks": {
    "configuration": { "status": "pass", "details": "ok" },
    "hardware": { "status": "pass", "details": "CUDA available=true, gpu=RTX 4090, memory=2.3/24.0GB" },
    "s3": { "status": "pass", "details": "configured=true" },
    "model": { "status": "pass", "details": "model_dir=/runpod-volume/..., present=true" }
  }
}
```

## Response Format

Success response:
```json
{
  "status": "completed",
  "filename": "550e8400-e29b-41d4-a716-446655440000.ogg",
  "url": "https://s3.../presigned-url",
  "s3_key": "550e8400-e29b-41d4-a716-446655440000.ogg",
  "metadata": {
    "sample_rate": 24000,
    "codec": "opus",
    "bitrate": "128k",
    "duration": 5.23,
    "device": "cuda",
    "model_repo": "OpenMOSS-Team/MOSS-TTS",
    "mode": "generation",
    "reference_audio": "speaker_demo.wav"
  }
}
```

Error response:
```json
{
  "error": "Text too long (max 10000 chars)",
  "error_type": "ValueError"
}
```

## API Reference

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize (max 10,000 characters) |
| `mode` | string | No | `generation` or `continuation` (default: `generation`) |
| `reference_audio` | string | No | Path or URL to reference audio for voice cloning |
| `prefix_audio` | string | Conditional | Required when `mode=continuation` |
| `expected_tokens` | int | No | Expected duration in tokens (for consistency) |
| `max_new_tokens` | int | No | Maximum tokens to generate (128-8192, default: 4096) |
| `audio_temperature` | float | No | Sampling temperature (0-5, default: 1.7) |
| `audio_top_p` | float | No | Nucleus sampling threshold (0-1, default: 0.8) |
| `audio_top_k` | int | No | Top-k sampling (1-200, default: 25) |
| `audio_repetition_penalty` | float | No | Repetition penalty (0.8-2.0, default: 1.0) |
| `enable_chunking` | bool | No | Enable text chunking (default: false) |
| `max_chars_per_chunk` | int | No | Characters per chunk (50-1000, default: 300) |
| `enable_crossfade` | bool | No | Enable crossfade between chunks (default: true) |
| `crossfade_ms` | int | No | Crossfade duration (0-2000, default: 140) |
| `session_id` | string | No | Custom session identifier (auto-generated if omitted) |

### Supported Audio Formats

**Input (reference/prefix audio):** `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.aac`, `.opus`

**Output:** `.ogg` (Opus codec, 128k VBR)

## Development

### Project Structure

```
Moss-TTS/
├── Dockerfile              # Container definition
├── bootstrap.sh            # First-boot setup script
├── config.py               # Configuration management
├── handler.py              # RunPod request handler
├── serverless_engine.py    # MOSS-TTS inference wrapper
├── requirements.txt        # Runtime dependencies
├── IMPLEMENTATION.md       # Implementation guide
├── README.md               # This file
└── docs/
    └── diagrams/
        ├── architecture.drawio
        └── data-flow.drawio
```

### First Boot Process

On initial container start, `bootstrap.sh`:

1. Clones `OpenMOSS/MOSS-TTS` source from GitHub
2. Creates Python virtual environment
3. Installs MOSS-TTS with CUDA 12.8 PyTorch extras
4. Installs serverless runtime (runpod, boto3, huggingface-hub)
5. Downloads model weights from HuggingFace
6. Copies handler/config/engine from image to network volume
7. Starts the RunPod serverless handler

Subsequent boots skip installation and use cached dependencies.

### Local Testing

```bash
# Warmup test (loads models and exits)
python handler.py --warmup

# Run with mock RunPod environment
python -c "
from handler import handler
from config import config

# Test health check
result = list(handler({'id': 'test', 'input': {'action': 'health_check'}}))
print(result)
"
```

## Troubleshooting

### Common Issues

**`ModuleNotFoundError` on first run**
- Check `bootstrap.log` in `/runpod-volume/moss-tts/`
- Verify virtual environment was created: `ls -la /runpod-volume/moss-tts/venv/`

**HuggingFace download failure**
- Verify `HF_TOKEN` is set correctly
- Check model repo access: `hf user whoami`
- Network volume may need more space

**OOM during generation**
- Reduce `max_new_tokens`
- Switch to local model: `MODEL_REPO=OpenMOSS-Team/MOSS-TTS-Local-Transformer`
- Use larger GPU class (24GB+ recommended)

**Slow cold start**
- Expected on first boot (~2-5 minutes)
- Warm-start by sending a tiny generation request after deployment
- Subsequent requests use cached model

**Invalid audio file errors**
- Verify file exists in `AUDIO_VOICES_DIR`
- Check file extension is supported
- Ensure no path traversal (`../`) in filenames

### Logs

- Bootstrap log: `/runpod-volume/moss-tts/bootstrap.log`
- RunPod worker logs: Available in RunPod console
- Python logs include timestamps and job IDs

## Performance Notes

- **Cold start:** 2-5 minutes (first boot includes model download)
- **Warm start:** 5-10 seconds (model already cached)
- **Generation time:** ~0.5-2x real-time depending on GPU and text length
- **GPU Memory:** 8-16GB VRAM typical usage

## Contributing

This project follows the implementation guide in [IMPLEMENTATION.md](./IMPLEMENTATION.md).

When contributing:
1. Follow existing code style
2. Add validation for new parameters
3. Update documentation
4. Test with both generation and continuation modes

## License

This project uses the MOSS-TTS model from [OpenMOSS-Team/MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS). Please refer to the model's license for usage terms.

## Acknowledgments

- [OpenMOSS Team](https://github.com/OpenMOSS) for the MOSS-TTS model
- [RunPod](https://www.runpod.io/) for serverless GPU infrastructure
- [IndexTTS](https://github.com/) for the serverless worker template pattern
