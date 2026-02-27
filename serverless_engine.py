# coding=utf-8
# MOSS-TTS RunPod Serverless Inference Engine

import gc
import importlib.util
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch

import config

from transformers import AutoModel, AutoProcessor

log = logging.getLogger(__name__)

# Disable broken cuDNN SDPA backend, matching upstream usage.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

_WHITESPACE_RE = re.compile(r"\s+")
_MODEL_INIT_LOCK = threading.Lock()


def _is_local_only_miss(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "couldn't connect to 'https://huggingface.co'" in text
        or "cannot find the requested files in the disk cache" in text
        or "local_files_only" in text
        or "localentrynotfounderror" in text
    )


def is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    """Split text at sentence/clause boundaries where possible."""
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    normalized = _WHITESPACE_RE.sub(" ", (text or "")).strip()
    if not normalized:
        return []

    if len(normalized) <= max_chars:
        return [normalized]

    sentence_enders = {".", "!", "?", "。", "！", "？"}
    clause_enders = {",", ";", ":", "，", "；", "："}
    closers = {'"', "'", ")", "]", "}"}

    chunks: List[str] = []
    remaining = normalized

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[: max_chars + 1]
        candidate_sentence = None
        candidate_clause = None
        candidate_space = None

        for idx in range(1, len(window)):
            if not window[idx].isspace():
                continue

            candidate_space = idx
            prev = window[idx - 1]
            prev2 = window[idx - 2] if idx >= 2 else ""

            if prev in sentence_enders or (prev in closers and prev2 in sentence_enders):
                candidate_sentence = idx
            elif prev in clause_enders or (prev in closers and prev2 in clause_enders):
                candidate_clause = idx

        split_at = candidate_sentence or candidate_clause or candidate_space
        if split_at is None:
            split_at = max_chars

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return chunks


def crossfade_chunks(audio_chunks: List[torch.Tensor], crossfade_ms: int, sample_rate: int) -> torch.Tensor:
    if not audio_chunks:
        return torch.tensor([], dtype=torch.float32)
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    crossfade_samples = int((crossfade_ms / 1000.0) * sample_rate)
    if crossfade_samples <= 0:
        return torch.cat(audio_chunks, dim=-1)

    result = audio_chunks[0]
    for chunk in audio_chunks[1:]:
        if len(result) < crossfade_samples or len(chunk) < crossfade_samples:
            result = torch.cat([result, chunk], dim=-1)
            continue

        fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=result.device)
        fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=chunk.device)

        result_tail = result[-crossfade_samples:] * fade_out
        chunk_head = chunk[:crossfade_samples] * fade_in
        blended = result_tail + chunk_head

        result = torch.cat([result[:-crossfade_samples], blended, chunk[crossfade_samples:]], dim=-1)

    return result


class MossTTSInference:
    """MOSS-TTS inference wrapper for RunPod serverless."""

    def __init__(
        self,
        model_repo: Optional[str] = None,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ):
        self.model_repo = model_repo or config.config.MODEL_REPO
        self.model_dir = Path(model_dir) if model_dir else config.config.MODEL_DIR
        self.device = device or config.config.device
        self.dtype = (dtype or config.config.default_dtype).lower()
        self.attn_implementation = attn_implementation or config.config.default_attn_implementation
        self.model_revision = config.config.MODEL_REVISION

        self._model = None
        self._processor = None
        self._torch_device = None
        self._torch_dtype = None
        self._sample_rate = config.DEFAULT_SAMPLE_RATE

    @staticmethod
    def _is_complete_model_dir(model_path: Path) -> bool:
        if not model_path.exists():
            return False
        has_config = (model_path / "config.json").exists()
        has_weights = (model_path / "model.safetensors").exists() or (
            model_path / "model.safetensors.index.json"
        ).exists()
        return has_config and has_weights

    def _find_runpod_cached_snapshot(self) -> Optional[Path]:
        cache_root = config.config.RUNPOD_HF_CACHE_DIR
        snapshots_dir = cache_root / f"models--{self.model_repo.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.exists():
            return None

        if self.model_revision:
            pinned = snapshots_dir / self.model_revision
            if self._is_complete_model_dir(pinned):
                return pinned

        candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for snapshot in candidates:
            if self._is_complete_model_dir(snapshot):
                return snapshot
        return None

    def _resolve_model_source(self) -> Tuple[str, bool, str]:
        runpod_snapshot = self._find_runpod_cached_snapshot()
        if runpod_snapshot is not None:
            return str(runpod_snapshot), True, "runpod-cache"
        if self._is_complete_model_dir(self.model_dir):
            return str(self.model_dir), True, "local-volume"
        return self.model_repo, False, "huggingface-hub"

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype in {"auto", ""}:
            return torch.bfloat16 if self._torch_device.type == "cuda" else torch.float32
        if self.dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float32

    def _resolve_attn_implementation(self) -> Optional[str]:
        requested = (self.attn_implementation or "").strip().lower()

        if requested == "none":
            return None
        if requested not in {"", "auto"}:
            return self.attn_implementation

        if (
            self._torch_device.type == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and self._torch_dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability(self._torch_device)
            if major >= 8:
                return "flash_attention_2"

        if self._torch_device.type == "cuda":
            return "sdpa"

        return "eager"

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with _MODEL_INIT_LOCK:
            if self._model is not None and self._processor is not None:
                return

            cuda_available = torch.cuda.is_available()
            requested_device = (self.device or "cpu").lower()
            if requested_device == "cuda" and not cuda_available:
                log.warning(
                    "CUDA requested but unavailable. Falling back to CPU. "
                    "This usually indicates a CUDA runtime/driver mismatch in the container."
                )

            self._torch_device = torch.device("cuda" if requested_device == "cuda" and cuda_available else "cpu")
            self._torch_dtype = self._resolve_dtype()
            model_source, local_files_only, source_kind = self._resolve_model_source()

            resolved_attn = self._resolve_attn_implementation()
            log.info(
                "Loading MOSS-TTS model from %s (source=%s, local_only=%s)",
                model_source,
                source_kind,
                local_files_only,
            )
            log.info("Device=%s dtype=%s attn=%s", self._torch_device, self._torch_dtype, resolved_attn)

            base_processor_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": local_files_only,
            }
            if self.model_revision and not local_files_only:
                base_processor_kwargs["revision"] = self.model_revision

            processor_kwargs = dict(base_processor_kwargs)
            try:
                self._processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
            except Exception as exc:
                if local_files_only and _is_local_only_miss(exc):
                    log.warning(
                        "Local-only processor load missed files from source=%s; retrying with online fallback.",
                        source_kind,
                    )
                    processor_kwargs["local_files_only"] = False
                    if self.model_revision:
                        processor_kwargs["revision"] = self.model_revision
                    self._processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
                else:
                    raise
            if hasattr(self._processor, "audio_tokenizer"):
                self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(self._torch_device)

            base_model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "dtype": self._torch_dtype,
                "local_files_only": local_files_only,
            }
            if self.model_revision and not local_files_only:
                base_model_kwargs["revision"] = self.model_revision
            if resolved_attn:
                base_model_kwargs["attn_implementation"] = resolved_attn

            model_kwargs = dict(base_model_kwargs)
            try:
                self._model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self._torch_device)
            except Exception as exc:
                if local_files_only and _is_local_only_miss(exc):
                    log.warning(
                        "Local-only model load missed files from source=%s; retrying with online fallback.",
                        source_kind,
                    )
                    model_kwargs["local_files_only"] = False
                    if self.model_revision:
                        model_kwargs["revision"] = self.model_revision
                    self._model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self._torch_device)
                else:
                    raise
            self._model.eval()

            self._sample_rate = int(getattr(self._processor.model_config, "sampling_rate", config.DEFAULT_SAMPLE_RATE))
            log.info("Model loaded; sample_rate=%s", self._sample_rate)

    def _resolve_voice_path_or_url(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None

        raw_value = value.strip()
        if not raw_value:
            return None

        if is_http_url(raw_value):
            return raw_value

        audio_root = config.config.AUDIO_VOICES_DIR.resolve()

        candidate = Path(raw_value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if not str(resolved).startswith(str(audio_root)):
                raise ValueError(f"Audio path must stay within {audio_root}")
            if not resolved.exists():
                raise ValueError(f"Audio file not found: {raw_value}")
            if resolved.suffix.lower() not in config.AUDIO_EXTS:
                raise ValueError(f"Unsupported audio extension: {resolved.suffix}")
            return str(resolved)

        direct = (audio_root / raw_value).resolve()
        if str(direct).startswith(str(audio_root)) and direct.exists() and direct.suffix.lower() in config.AUDIO_EXTS:
            return str(direct)

        if Path(raw_value).suffix:
            raise ValueError(f"Audio file not found: {raw_value}")

        for ext in config.AUDIO_EXTS:
            test_path = (audio_root / f"{raw_value}{ext}").resolve()
            if str(test_path).startswith(str(audio_root)) and test_path.exists():
                return str(test_path)

        available = [f.name for f in audio_root.glob("*") if f.suffix.lower() in config.AUDIO_EXTS]
        raise ValueError(f"Audio file '{raw_value}' not found. Available: {available}")

    def _build_conversation(
        self,
        text: str,
        mode: str,
        reference_audio: Optional[str],
        prefix_audio: Optional[str],
        expected_tokens: Optional[int],
    ) -> Tuple[List[List[Any]], str]:
        user_kwargs: Dict[str, Any] = {"text": text}
        if expected_tokens is not None:
            user_kwargs["tokens"] = int(expected_tokens)

        if reference_audio:
            user_kwargs["reference"] = [reference_audio]

        if mode == "continuation":
            if not prefix_audio:
                raise ValueError("prefix_audio is required for continuation mode")
            conversation = [
                self._processor.build_user_message(**user_kwargs),
                self._processor.build_assistant_message(audio_codes_list=[prefix_audio]),
            ]
            return [conversation], "continuation"

        conversation = [self._processor.build_user_message(**user_kwargs)]
        return [conversation], "generation"

    def _generate_single(
        self,
        text: str,
        mode: str,
        reference_audio: Optional[str],
        prefix_audio: Optional[str],
        expected_tokens: Optional[int],
        max_new_tokens: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
    ) -> Tuple[torch.Tensor, int]:
        conversations, processor_mode = self._build_conversation(
            text=text,
            mode=mode,
            reference_audio=reference_audio,
            prefix_audio=prefix_audio,
            expected_tokens=expected_tokens,
        )

        batch = self._processor(conversations, mode=processor_mode)
        input_ids = batch["input_ids"].to(self._torch_device)
        attention_mask = batch["attention_mask"].to(self._torch_device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
            )

        messages = self._processor.decode(outputs)
        if not messages:
            raise RuntimeError("Model returned no messages")

        audio = messages[0].audio_codes_list[0]
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.detach().float().to(self._torch_device)
        else:
            audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=self._torch_device)

        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.reshape(-1)

        return audio_tensor, self._sample_rate

    def generate_audio(
        self,
        text: str,
        mode: str = "generation",
        reference_audio: Optional[str] = None,
        prefix_audio: Optional[str] = None,
        expected_tokens: Optional[int] = None,
        max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
        audio_temperature: float = config.DEFAULT_AUDIO_TEMPERATURE,
        audio_top_p: float = config.DEFAULT_AUDIO_TOP_P,
        audio_top_k: int = config.DEFAULT_AUDIO_TOP_K,
        audio_repetition_penalty: float = config.DEFAULT_AUDIO_REPETITION_PENALTY,
        enable_chunking: bool = False,
        max_chars_per_chunk: int = 300,
        enable_crossfade: bool = True,
        crossfade_ms: int = 140,
    ) -> Tuple[torch.Tensor, int]:
        self._load_model()

        resolved_reference = self._resolve_voice_path_or_url(reference_audio)
        resolved_prefix = self._resolve_voice_path_or_url(prefix_audio)

        if enable_chunking and mode == "generation" and len(text) > max_chars_per_chunk:
            chunks = chunk_text(text, max_chars=max_chars_per_chunk)
            if not chunks:
                raise ValueError("Text is empty after normalization")

            generated: List[torch.Tensor] = []
            for idx, chunk in enumerate(chunks, start=1):
                log.info("Generating chunk %s/%s", idx, len(chunks))
                chunk_audio, _ = self._generate_single(
                    text=chunk,
                    mode=mode,
                    reference_audio=resolved_reference,
                    prefix_audio=resolved_prefix,
                    expected_tokens=expected_tokens,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=audio_repetition_penalty,
                )
                generated.append(chunk_audio)

            if enable_crossfade and len(generated) > 1:
                return crossfade_chunks(generated, crossfade_ms=crossfade_ms, sample_rate=self._sample_rate), self._sample_rate
            return torch.cat(generated, dim=-1), self._sample_rate

        return self._generate_single(
            text=text,
            mode=mode,
            reference_audio=resolved_reference,
            prefix_audio=resolved_prefix,
            expected_tokens=expected_tokens,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )

    def generate_audio_stream_decoded(
        self,
        text: str,
        mode: str = "generation",
        reference_audio: Optional[str] = None,
        prefix_audio: Optional[str] = None,
        expected_tokens: Optional[int] = None,
        max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
        audio_temperature: float = config.DEFAULT_AUDIO_TEMPERATURE,
        audio_top_p: float = config.DEFAULT_AUDIO_TOP_P,
        audio_top_k: int = config.DEFAULT_AUDIO_TOP_K,
        audio_repetition_penalty: float = config.DEFAULT_AUDIO_REPETITION_PENALTY,
        max_chars_per_chunk: int = 150,
        enable_crossfade: bool = True,
        crossfade_ms: int = 100,
        chunk_pause_ms: int = 300,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate streaming audio chunks as base64-encoded signed int16 PCM.

        Each text chunk is synthesized and yielded immediately so the client
        receives audio as soon as the first chunk finishes inference, rather
        than waiting for all chunks to complete.

        When crossfade is enabled, only the last `crossfade_ms` of each chunk
        is held back and blended with the start of the next chunk.

        Yields:
            Dictionaries with streaming chunk data:
            - {"status": "streaming", "chunk": int, "format": "pcm_16",
               "audio_chunk": base64_pcm_bytes, "sample_rate": int}
            - {"status": "complete", "format": "pcm_16", "total_chunks": int}
        """
        import base64
        import traceback

        self._load_model()

        resolved_reference = self._resolve_voice_path_or_url(reference_audio)
        resolved_prefix = self._resolve_voice_path_or_url(prefix_audio)

        try:
            chunks = chunk_text(text, max_chars=max_chars_per_chunk) if max_chars_per_chunk and max_chars_per_chunk > 0 else [text]
            if not chunks:
                yield {"error": "Text is empty after normalization"}
                return

            output_sample_rate = None
            crossfade_samples = 0
            pause_samples = 0
            crossfade_tail = None
            chunk_num = 0

            def _to_int16_bytes(tensor: torch.Tensor) -> str:
                """Convert float tensor to base64-encoded int16 PCM."""
                arr = tensor.detach().cpu().numpy()
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = np.clip(arr, -1.0, 1.0)
                    arr = (arr * 32767).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
                return base64.b64encode(arr.tobytes()).decode("utf-8")

            for i, chunk_text_content in enumerate(chunks):
                log.info("Streaming chunk %s/%s", i + 1, len(chunks))

                audio_tensor, _ = self._generate_single(
                    text=chunk_text_content,
                    mode=mode,
                    reference_audio=resolved_reference,
                    prefix_audio=resolved_prefix,
                    expected_tokens=expected_tokens,
                    max_new_tokens=max_new_tokens,
                    audio_temperature=audio_temperature,
                    audio_top_p=audio_top_p,
                    audio_top_k=audio_top_k,
                    audio_repetition_penalty=audio_repetition_penalty,
                )

                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.reshape(-1)

                if output_sample_rate is None:
                    output_sample_rate = self._sample_rate
                    crossfade_samples = (
                        int(output_sample_rate * (float(crossfade_ms) / 1000.0))
                        if crossfade_ms and enable_crossfade
                        else 0
                    )
                    pause_samples = int(output_sample_rate * (float(chunk_pause_ms) / 1000.0)) if chunk_pause_ms > 0 else 0
                    log.info("Model output sample rate: %s", output_sample_rate)

                is_last_chunk = (i == len(chunks) - 1)

                # Blend crossfade tail from previous chunk with start of this chunk
                if crossfade_tail is not None:
                    if crossfade_samples > 0:
                        cf = min(crossfade_samples, len(crossfade_tail), len(audio_tensor))
                        if cf > 0:
                            fade_out = torch.linspace(1.0, 0.0, cf, device=audio_tensor.device, dtype=audio_tensor.dtype)
                            fade_in = 1.0 - fade_out
                            blended = crossfade_tail * fade_out + audio_tensor[:cf] * fade_in
                            audio_to_emit = torch.cat([blended, audio_tensor[cf:]], dim=-1)
                        else:
                            audio_to_emit = torch.cat([crossfade_tail, audio_tensor], dim=-1)
                    else:
                        audio_to_emit = torch.cat([crossfade_tail, audio_tensor], dim=-1)
                else:
                    audio_to_emit = audio_tensor

                # Hold back crossfade overlap for next chunk (unless last chunk)
                if crossfade_samples > 0 and not is_last_chunk and len(audio_to_emit) > crossfade_samples:
                    crossfade_tail = audio_to_emit[-crossfade_samples:]
                    audio_to_emit = audio_to_emit[:-crossfade_samples]
                else:
                    crossfade_tail = None

                # Append silence gap after non-last chunks for natural pacing
                if not is_last_chunk and pause_samples > 0:
                    silence = torch.zeros(pause_samples, device=audio_to_emit.device, dtype=audio_to_emit.dtype)
                    audio_to_emit = torch.cat([audio_to_emit, silence], dim=-1)

                # Yield this chunk's audio immediately
                if audio_to_emit is not None and len(audio_to_emit) > 0:
                    chunk_num += 1
                    yield {
                        "status": "streaming",
                        "chunk": chunk_num,
                        "format": "pcm_16",
                        "audio_chunk": _to_int16_bytes(audio_to_emit),
                        "sample_rate": output_sample_rate,
                    }

            # Flush any remaining crossfade tail
            if crossfade_tail is not None and len(crossfade_tail) > 0:
                chunk_num += 1
                yield {
                    "status": "streaming",
                    "chunk": chunk_num,
                    "format": "pcm_16",
                    "audio_chunk": _to_int16_bytes(crossfade_tail),
                    "sample_rate": output_sample_rate or config.DEFAULT_SAMPLE_RATE,
                }

            yield {
                "status": "complete",
                "format": "pcm_16",
                "message": "All chunks streamed",
                "total_chunks": chunk_num,
            }

        except Exception as exc:
            error_trace = traceback.format_exc()
            log.error("Streaming mode failed: %s", exc)
            log.error("Traceback: %s", error_trace)
            yield {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": error_trace
            }

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if self._torch_device is not None and self._torch_device.type == "cuda":
            torch.cuda.empty_cache()


_inference_engine: Optional[MossTTSInference] = None


def get_inference_engine(
    model_repo: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> MossTTSInference:
    global _inference_engine

    if _inference_engine is None:
        _inference_engine = MossTTSInference(
            model_repo=model_repo,
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

    return _inference_engine
