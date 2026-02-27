# coding=utf-8
# MOSS-TTS RunPod Serverless Handler

import argparse
import os
import subprocess
import tempfile
import time
import traceback
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import runpod
import torch
import torchaudio
from botocore.config import Config as BotoConfig

import config as config_module
from config import config
from serverless_engine import get_inference_engine

log = runpod.RunPodLogger()


def is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
    return default


def cleanup_old_files(directory: str, days: int = 2) -> None:
    try:
        from pathlib import Path

        output_dir = Path(directory)
        if not output_dir.exists():
            return

        cutoff = time.time() - (days * 24 * 60 * 60)
        removed = 0

        for file_path in output_dir.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                removed += 1

        if removed > 0:
            log.info(f"Removed {removed} expired files from {directory}")
    except Exception as exc:
        log.error(f"Cleanup failed: {exc}")


def get_s3_client():
    missing = []
    if not config.S3_ENDPOINT_URL:
        missing.append("S3_ENDPOINT_URL")
    if not config.S3_ACCESS_KEY_ID:
        missing.append("S3_ACCESS_KEY_ID")
    if not config.S3_SECRET_ACCESS_KEY:
        missing.append("S3_SECRET_ACCESS_KEY")
    if not config.S3_BUCKET_NAME:
        missing.append("S3_BUCKET_NAME")

    if missing:
        raise RuntimeError(f"Missing S3 configuration: {', '.join(missing)}")

    endpoint_host = urlparse(config.S3_ENDPOINT_URL).hostname or ""
    region = config.S3_REGION
    if endpoint_host.endswith(".backblazeb2.com"):
        parts = endpoint_host.split(".")
        if len(parts) >= 4 and parts[0] == "s3":
            inferred_region = parts[1]
            if region in {"", "us-east-1"}:
                region = inferred_region
                log.info(
                    f"Inferred S3 region '{region}' from endpoint host '{endpoint_host}' "
                    f"(configured region was '{config.S3_REGION}')"
                )

    client_cfg = BotoConfig(
        signature_version=config.S3_SIGNATURE_VERSION,
        s3={"addressing_style": config.S3_ADDRESSING_STYLE},
    )

    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT_URL,
        region_name=region,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        config=client_cfg,
    )


def encode_to_opus(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
        tmp_ogg_path = tmp_ogg.name

    try:
        if audio_tensor is None:
            raise RuntimeError("Audio tensor is None")

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        torchaudio.save(tmp_wav_path, audio_tensor, sample_rate)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", tmp_wav_path,
            "-ar", "24000",
            "-c:a", "libopus",
            "-b:a", "128k",
            "-vbr", "on",
            "-compression_level", "10",
            tmp_ogg_path,
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        with open(tmp_ogg_path, "rb") as file_obj:
            return file_obj.read()

    finally:
        for tmp_file in [tmp_wav_path, tmp_ogg_path]:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            except OSError:
                pass


def upload_to_s3(audio_bytes: bytes, filename: str) -> str:
    s3 = get_s3_client()
    s3.put_object(
        Bucket=config.S3_BUCKET_NAME,
        Key=filename,
        Body=audio_bytes,
        ContentType="audio/ogg; codecs=opus",
    )

    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.S3_BUCKET_NAME, "Key": filename},
        ExpiresIn=3600,
    )


def _enforce_audio_duration_limit(audio_path, field_name: str, max_seconds: Optional[int]) -> None:
    if max_seconds is None or int(max_seconds) <= 0:
        return
    try:
        info = torchaudio.info(str(audio_path))
        if info.sample_rate <= 0:
            return
        duration_sec = float(info.num_frames) / float(info.sample_rate)
        if duration_sec > float(max_seconds):
            raise ValueError(
                f"{field_name} too long ({duration_sec:.1f}s). "
                f"Max supported is {int(max_seconds)}s on this worker profile."
            )
    except ValueError:
        raise
    except Exception as exc:
        log.warning(f"Could not inspect {field_name} duration for '{audio_path}': {exc}")


def _validate_audio_field(value: Optional[str], field_name: str, max_seconds: Optional[int] = None) -> Optional[str]:
    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    if is_http_url(value):
        return value

    audio_root = config.AUDIO_VOICES_DIR.resolve()

    candidate = (audio_root / value).resolve()
    if not str(candidate).startswith(str(audio_root)):
        raise ValueError(f"Invalid {field_name} path")

    if candidate.exists():
        if candidate.suffix.lower() not in config.AUDIO_EXTS:
            raise ValueError(f"Unsupported {field_name} extension: {candidate.suffix}")
        _enforce_audio_duration_limit(candidate, field_name, max_seconds)
        return value

    if os.path.splitext(value)[1]:
        raise ValueError(f"{field_name} '{value}' not found")

    for ext in config.AUDIO_EXTS:
        test_path = audio_root / f"{value}{ext}"
        if test_path.exists():
            _enforce_audio_duration_limit(test_path, field_name, max_seconds)
            return f"{value}{ext}"

    available = [
        f.name for f in audio_root.glob("*")
        if f.suffix.lower() in config.AUDIO_EXTS
    ]
    raise ValueError(f"{field_name} '{value}' not found. Available: {available}")


def health_check() -> Dict[str, Any]:
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {},
    }

    config_valid = config.validate()
    status["checks"]["configuration"] = {
        "status": "pass" if config_valid else "fail",
        "details": "ok" if config_valid else "; ".join(config.validation_errors),
    }

    cuda_ok = torch.cuda.is_available()
    details = f"CUDA available={cuda_ok}, configured_device={config.device}"
    if cuda_ok:
        gpu = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        details += f", gpu={gpu}, memory={alloc:.1f}/{total:.1f}GB"

    status["checks"]["hardware"] = {
        "status": "pass" if cuda_ok else "warn",
        "details": details,
    }

    s3_ready = all([
        config.S3_ENDPOINT_URL,
        config.S3_ACCESS_KEY_ID,
        config.S3_SECRET_ACCESS_KEY,
        config.S3_BUCKET_NAME,
    ])
    status["checks"]["s3"] = {
        "status": "pass" if s3_ready else "fail",
        "details": f"configured={s3_ready}",
    }

    model_present = config.MODEL_DIR.exists() and any(config.MODEL_DIR.iterdir())
    status["checks"]["model"] = {
        "status": "pass" if model_present else "warn",
        "details": f"model_dir={config.MODEL_DIR}, present={model_present}",
    }

    all_pass = all(check["status"] == "pass" for check in status["checks"].values())
    status["status"] = "healthy" if all_pass else "unhealthy"
    return status


def extract_and_validate_params(job_input: Dict[str, Any]):
    text = job_input.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, {"error": "Missing or empty 'text' parameter"}
    if len(text) > 10000:
        return None, {"error": "Text too long (max 10000 chars)"}

    mode = str(job_input.get("mode", "generation")).strip().lower()
    if mode not in {"generation", "continuation"}:
        return None, {"error": "mode must be 'generation' or 'continuation'"}

    try:
        reference_audio = _validate_audio_field(
            job_input.get("reference_audio"),
            "reference_audio",
            config.max_reference_audio_seconds,
        )
        prefix_audio = _validate_audio_field(
            job_input.get("prefix_audio"),
            "prefix_audio",
            config.max_prefix_audio_seconds,
        )
    except ValueError as exc:
        return None, {"error": str(exc)}

    if mode == "continuation" and not prefix_audio:
        return None, {"error": "prefix_audio is required when mode='continuation'"}

    expected_tokens = job_input.get("expected_tokens")
    if expected_tokens is not None:
        try:
            expected_tokens = int(expected_tokens)
            if expected_tokens < 1:
                return None, {"error": "expected_tokens must be >= 1"}
        except Exception:
            return None, {"error": "expected_tokens must be an integer"}

    try:
        max_new_tokens = int(job_input.get("max_new_tokens", config.default_max_new_tokens))
        if max_new_tokens < 128 or max_new_tokens > 8192:
            return None, {"error": "max_new_tokens must be between 128 and 8192"}

        audio_temperature = float(job_input.get("audio_temperature", config.default_audio_temperature))
        audio_top_p = float(job_input.get("audio_top_p", config.default_audio_top_p))
        audio_top_k = int(job_input.get("audio_top_k", config.default_audio_top_k))
        audio_repetition_penalty = float(
            job_input.get("audio_repetition_penalty", config.default_audio_repetition_penalty)
        )
    except Exception:
        return None, {"error": "Invalid decoding parameters"}

    if audio_temperature <= 0.0 or audio_temperature > 5.0:
        return None, {"error": "audio_temperature must be in (0, 5]"}
    if audio_top_p <= 0.0 or audio_top_p > 1.0:
        return None, {"error": "audio_top_p must be in (0, 1]"}
    if audio_top_k < 1 or audio_top_k > 200:
        return None, {"error": "audio_top_k must be in [1, 200]"}
    if audio_repetition_penalty < 0.8 or audio_repetition_penalty > 2.0:
        return None, {"error": "audio_repetition_penalty must be in [0.8, 2.0]"}

    try:
        enable_chunking = parse_bool(job_input.get("enable_chunking"), config.default_enable_chunking)
        max_chars_per_chunk = int(job_input.get("max_chars_per_chunk", config.default_max_chars_per_chunk))
        enable_crossfade = parse_bool(job_input.get("enable_crossfade"), config.default_enable_crossfade)
        crossfade_ms = int(job_input.get("crossfade_ms", config.default_crossfade_ms))
    except Exception:
        return None, {"error": "Invalid chunking or crossfade parameters"}

    if max_chars_per_chunk < 50 or max_chars_per_chunk > 1000:
        return None, {"error": "max_chars_per_chunk must be between 50 and 1000"}
    if crossfade_ms < 0 or crossfade_ms > 2000:
        return None, {"error": "crossfade_ms must be between 0 and 2000"}

    # Streaming parameters
    stream = parse_bool(job_input.get("stream"), False)
    output_format = job_input.get("output_format", "pcm_16")
    stream_max_chars_per_chunk = job_input.get("stream_max_chars_per_chunk")
    stream_crossfade_ms = job_input.get("stream_crossfade_ms")
    chunk_pause_ms = int(job_input.get("chunk_pause_ms", config.default_chunk_pause_ms))

    if stream_max_chars_per_chunk is not None:
        try:
            stream_max_chars_per_chunk = int(stream_max_chars_per_chunk)
            if stream_max_chars_per_chunk < 50 or stream_max_chars_per_chunk > 1000:
                return None, {"error": "stream_max_chars_per_chunk must be between 50 and 1000"}
        except Exception:
            return None, {"error": "stream_max_chars_per_chunk must be an integer"}

    if stream_crossfade_ms is not None:
        try:
            stream_crossfade_ms = int(stream_crossfade_ms)
            if stream_crossfade_ms < 0 or stream_crossfade_ms > 2000:
                return None, {"error": "stream_crossfade_ms must be between 0 and 2000"}
        except Exception:
            return None, {"error": "stream_crossfade_ms must be an integer"}

    if chunk_pause_ms < 0 or chunk_pause_ms > 2000:
        return None, {"error": "chunk_pause_ms must be between 0 and 2000"}

    if output_format != "pcm_16":
        return None, {"error": "Invalid output_format. Only 'pcm_16' is currently supported"}

    session_id = str(job_input.get("session_id") or uuid4())

    return {
        "text": text,
        "mode": mode,
        "reference_audio": reference_audio,
        "prefix_audio": prefix_audio,
        "expected_tokens": expected_tokens,
        "max_new_tokens": max_new_tokens,
        "audio_temperature": audio_temperature,
        "audio_top_p": audio_top_p,
        "audio_top_k": audio_top_k,
        "audio_repetition_penalty": audio_repetition_penalty,
        "enable_chunking": enable_chunking,
        "max_chars_per_chunk": max_chars_per_chunk,
        "enable_crossfade": enable_crossfade,
        "crossfade_ms": crossfade_ms,
        "stream": stream,
        "output_format": output_format,
        "stream_max_chars_per_chunk": stream_max_chars_per_chunk,
        "stream_crossfade_ms": stream_crossfade_ms,
        "chunk_pause_ms": chunk_pause_ms,
        "session_id": session_id,
    }, None


def handler_batch(job_input: Dict[str, Any]) -> Dict[str, Any]:
    cleanup_old_files(str(config.OUTPUT_AUDIO_DIR), days=config_module.CLEANUP_DAYS)

    params, error = extract_and_validate_params(job_input)
    if error:
        return error

    try:
        inference_engine = get_inference_engine()

        audio_out, sample_rate = inference_engine.generate_audio(
            text=params["text"],
            mode=params["mode"],
            reference_audio=params["reference_audio"],
            prefix_audio=params["prefix_audio"],
            expected_tokens=params["expected_tokens"],
            max_new_tokens=params["max_new_tokens"],
            audio_temperature=params["audio_temperature"],
            audio_top_p=params["audio_top_p"],
            audio_top_k=params["audio_top_k"],
            audio_repetition_penalty=params["audio_repetition_penalty"],
            enable_chunking=params["enable_chunking"],
            max_chars_per_chunk=params["max_chars_per_chunk"],
            enable_crossfade=params["enable_crossfade"],
            crossfade_ms=params["crossfade_ms"],
        )

        if audio_out is None or len(audio_out) == 0:
            return {"error": "No audio generated"}

        if audio_out.dim() > 1:
            audio_out = audio_out.reshape(-1)

        duration_seconds = len(audio_out) / float(sample_rate)
        audio_bytes = encode_to_opus(audio_out.cpu(), sample_rate)

        filename = f"{params['session_id']}.ogg"
        url = upload_to_s3(audio_bytes, filename)

        return {
            "status": "completed",
            "filename": filename,
            "url": url,
            "s3_key": filename,
            "metadata": {
                "sample_rate": 24000,
                "codec": "opus",
                "bitrate": "128k",
                "duration": duration_seconds,
                "device": config.device,
                "model_repo": config.MODEL_REPO,
                "mode": params["mode"],
                "reference_audio": params["reference_audio"],
            },
        }

    except Exception as exc:
        trace = traceback.format_exc()
        log.error(f"Batch generation failed: {exc}")
        log.error(trace)
        return {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": trace,
        }


def handler_stream(job_input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming mode handler - yields base64 PCM chunks as they're generated.

    Args:
        job_input: Job input dictionary

    Yields:
        Dictionaries with streaming chunk data
    """
    params, error = extract_and_validate_params(job_input)
    if error:
        log.error(f"Parameter validation failed: {error}")
        yield error
        return

    try:
        inference_engine = get_inference_engine()
        yield from inference_engine.generate_audio_stream_decoded(
            text=params["text"],
            mode=params["mode"],
            reference_audio=params["reference_audio"],
            prefix_audio=params["prefix_audio"],
            expected_tokens=params["expected_tokens"],
            max_new_tokens=params["max_new_tokens"],
            audio_temperature=params["audio_temperature"],
            audio_top_p=params["audio_top_p"],
            audio_top_k=params["audio_top_k"],
            audio_repetition_penalty=params["audio_repetition_penalty"],
            max_chars_per_chunk=params["stream_max_chars_per_chunk"] or params["max_chars_per_chunk"],
            enable_crossfade=params["enable_crossfade"],
            crossfade_ms=params["stream_crossfade_ms"] if params["stream_crossfade_ms"] is not None else params["crossfade_ms"],
            chunk_pause_ms=params["chunk_pause_ms"],
        )
    except Exception as exc:
        trace = traceback.format_exc()
        log.error(f"Streaming mode failed: {exc}")
        log.error(trace)
        yield {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": trace,
        }


def handler(job: Dict[str, Any]):
    job_id = job.get("id")
    job_input = job.get("input", {})

    if job_input.get("action") == "health_check":
        yield health_check()
        return

    stream = job_input.get("stream", False)
    output_format = job_input.get("output_format", "pcm_16")

    if stream:
        log.info(f"[{job_id}] Streaming mode: format={output_format}")
        yield from handler_stream(job_input)
        return

    # Batch mode - generate and upload
    log.info(f"[{job_id}] Batch mode - input keys: {list(job_input.keys())}")
    result = handler_batch(job_input)
    log.info(f"[{job_id}] Batch mode result status: {result.get('status', result.get('error', 'unknown'))}")
    yield result


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod handler for MOSS-TTS")
    parser.add_argument("--warmup", action="store_true", help="Load models and exit")
    args, _ = parser.parse_known_args()

    print("=== MOSS-TTS RunPod Handler Starting ===")
    print(f"Configured device: {config.device}")
    print(f"Working directory: {os.getcwd()}")

    if args.warmup:
        try:
            engine = get_inference_engine()
            engine._load_model()
            print("Warmup complete")
        except Exception as exc:
            print(f"Warmup failed: {exc}")
            traceback.print_exc()
            raise
        return

    if not config.validate():
        print("WARNING: Configuration validation errors detected")
        for err in config.validation_errors:
            print(f" - {err}")

    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
    })


if __name__ == "__main__":
    main()
