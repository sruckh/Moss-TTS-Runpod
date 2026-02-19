# coding=utf-8
# MOSS-TTS RunPod Serverless Inference Engine

import gc
import importlib.util
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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

        self._model = None
        self._processor = None
        self._torch_device = None
        self._torch_dtype = None
        self._sample_rate = config.DEFAULT_SAMPLE_RATE

    def _resolve_model_source(self) -> str:
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            return str(self.model_dir)
        return self.model_repo

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

        self._torch_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._torch_dtype = self._resolve_dtype()
        model_source = self._resolve_model_source()

        resolved_attn = self._resolve_attn_implementation()
        log.info("Loading MOSS-TTS model from %s", model_source)
        log.info("Device=%s dtype=%s attn=%s", self._torch_device, self._torch_dtype, resolved_attn)

        self._processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
        if hasattr(self._processor, "audio_tokenizer"):
            self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(self._torch_device)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self._torch_dtype,
        }
        if resolved_attn:
            model_kwargs["attn_implementation"] = resolved_attn

        self._model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self._torch_device)
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

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
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
