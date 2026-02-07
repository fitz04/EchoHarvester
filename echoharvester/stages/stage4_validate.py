"""Stage 4: ASR Validation - Re-transcription and CER filtering."""

import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import torch

from echoharvester.config import Config
from echoharvester.db import Database, Status
from echoharvester.stages.base import BaseStage
from echoharvester.utils import (
    calculate_cer,
    is_acceptable_cer,
    normalize_for_cer,
    normalize_text,
)

logger = logging.getLogger(__name__)


class QwenTranscriber:
    """Wrapper for Qwen3-ASR transcription."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "ko",
    ):
        self.model_name = model_name
        self.language = language
        self._model = None

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if compute_type in ("auto", "bfloat16"):
            self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        elif compute_type == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        logger.info(f"Qwen ASR device: {self.device}, dtype: {self.dtype}")

    def _load_model(self):
        """Lazy load the Qwen ASR model."""
        if self._model is not None:
            return

        from qwen_asr import Qwen3ASRModel

        logger.info(f"Loading Qwen ASR model: {self.model_name}")
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device,
        )
        logger.info("Qwen ASR model loaded")

    def transcribe(self, audio_path: str | Path, **kwargs) -> str:
        """Transcribe a single audio file."""
        self._load_model()
        lang_map = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        results = self._model.transcribe(
            audio=str(audio_path),
            language=lang_map.get(self.language),
        )
        return results[0].text.strip() if results else ""

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
        **kwargs,
    ) -> list[str]:
        """Transcribe multiple audio files (native batch support)."""
        self._load_model()
        lang_map = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        results = []
        try:
            batch_results = self._model.transcribe(
                audio=[str(p) for p in audio_paths],
                language=lang_map.get(self.language),
            )
            for r in batch_results:
                results.append(r.text.strip() if r else "")
        except Exception as e:
            logger.error(f"Batch transcription error, falling back to sequential: {e}")
            for audio_path in audio_paths:
                try:
                    results.append(self.transcribe(audio_path))
                except Exception as e2:
                    logger.error(f"Transcription error for {audio_path}: {e2}")
                    results.append("")
        return results


class WhisperTranscriber:
    """Wrapper for faster-whisper transcription."""

    def __init__(
        self,
        model_name: str = "Systran/faster-whisper-medium",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "ko",
    ):
        self.model_name = model_name
        self.language = language
        self._model = None

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        logger.info(f"Whisper device: {self.device}, compute_type: {self.compute_type}")

    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {self.model_name}")
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type,
        )
        logger.info("Whisper model loaded")

    def transcribe(self, audio_path: str | Path, beam_size: int = 5) -> str:
        """Transcribe a single audio file."""
        self._load_model()
        segments, _ = self._model.transcribe(
            str(audio_path), beam_size=beam_size, language=self.language, vad_filter=False,
        )
        return " ".join(segment.text for segment in segments).strip()

    def transcribe_batch(
        self, audio_paths: list[str | Path], beam_size: int = 5,
    ) -> list[str]:
        """Transcribe multiple audio files (sequential, model kept loaded)."""
        self._load_model()
        results = []
        for audio_path in audio_paths:
            try:
                results.append(self.transcribe(audio_path, beam_size))
            except Exception as e:
                logger.error(f"Transcription error for {audio_path}: {e}")
                results.append("")
        return results


def create_transcriber(config: Config):
    """Factory: create transcriber based on config backend."""
    backend = config.validation.backend
    device = config.validation.resolve_device()
    compute_type = config.validation.resolve_compute_type()

    if backend == "qwen-asr":
        return QwenTranscriber(
            model_name=config.validation.model, device=device,
            compute_type=compute_type, language=config.validation.language,
        )
    else:
        return WhisperTranscriber(
            model_name=config.validation.model, device=device,
            compute_type=compute_type, language=config.validation.language,
        )


class ValidateStage(BaseStage):
    """Validation stage using ASR re-transcription."""

    name = "validate"

    def __init__(self, config: Config, db: Database):
        super().__init__(config, db)
        self.transcriber = None
        self.batch_size = config.validation.batch_size

    def _init_transcriber(self):
        """Initialize the transcriber with retry on failure."""
        if self.transcriber is not None:
            return

        for attempt in range(1, 4):
            try:
                self.transcriber = create_transcriber(self.config)
                return
            except Exception as e:
                if attempt == 3:
                    logger.error(f"Failed to initialize transcriber after 3 attempts: {e}")
                    raise
                logger.warning(
                    f"Transcriber init attempt {attempt}/3 failed: {e}. Retrying..."
                )
                import gc
                gc.collect()
                time.sleep(2 * attempt)

    async def process(self) -> AsyncGenerator[dict[str, Any], None]:
        """Process CPU-passed segments through GPU validation."""
        # Get segments ready for GPU validation
        segments = await self.db.get_segments_by_status(
            [Status.CPU_PASS.value],
            limit=10000,  # Process in larger batches
        )
        total = len(segments)

        if total == 0:
            logger.info("No segments to validate")
            return

        total_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info(
            f"[Stage4] Validating {total} segments in ~{total_batches} batches "
            f"(batch_size={self.batch_size})"
        )
        self._init_transcriber()

        # Process in batches
        batch_size = self.batch_size
        processed = 0
        total_pass = 0
        total_reject = 0
        total_error = 0
        all_cers: list[float] = []
        stage_start = time.time()

        for i in range(0, total, batch_size):
            if self._cancelled:
                break

            batch_num = i // batch_size + 1
            batch = segments[i : i + batch_size]
            audio_paths = [seg["audio_path"] for seg in batch]

            try:
                # Mark as processing
                segment_ids = [seg["id"] for seg in batch]
                await self.db.bulk_update_segments(
                    segment_ids, Status.GPU_PROCESSING.value
                )

                batch_start = time.time()

                # Transcribe batch
                transcriptions = self.transcriber.transcribe_batch(
                    audio_paths,
                    beam_size=self.config.validation.beam_size,
                )

                batch_elapsed = time.time() - batch_start

                # Calculate CER and update status
                batch_cers: list[float] = []
                batch_passed = 0
                batch_rejected = 0

                for seg, whisper_text in zip(batch, transcriptions):
                    seg_id = seg["id"]
                    normalized_text = seg.get("normalized_text", "")
                    original_text = seg.get("original_text", "")
                    subtitle_type = seg.get("subtitle_type", "unknown")

                    # Consistency check: ensure normalized_text matches original_text
                    if original_text:
                        expected_norm = normalize_text(original_text)
                        if normalized_text != expected_norm:
                            logger.warning(
                                f"[Stage4] {seg_id}: normalized_text mismatch, "
                                f"recalculating from original_text"
                            )
                            normalized_text = expected_norm
                            await self.db.update_segment(
                                seg_id, normalized_text=normalized_text
                            )

                    # Reject if ASR returned empty text
                    if not whisper_text or not whisper_text.strip():
                        batch_rejected += 1
                        total_reject += 1
                        await self.db.update_segment(
                            seg_id,
                            status=Status.GPU_REJECT.value,
                            whisper_text=whisper_text or "",
                            cer=None,
                            reject_reason="empty_asr_text",
                        )
                        processed += 1
                        continue

                    # Calculate CER
                    cer = calculate_cer(normalized_text, whisper_text)
                    batch_cers.append(cer)
                    all_cers.append(cer)

                    # Determine threshold based on subtitle type
                    if subtitle_type == "manual":
                        threshold = self.config.filters.cer_threshold_manual
                    else:
                        threshold = self.config.filters.cer_threshold_auto

                    # Check if acceptable
                    passed = is_acceptable_cer(
                        cer,
                        subtitle_type,
                        manual_threshold=self.config.filters.cer_threshold_manual,
                        auto_threshold=self.config.filters.cer_threshold_auto,
                    )

                    if passed:
                        batch_passed += 1
                        total_pass += 1
                        await self.db.update_segment(
                            seg_id,
                            status=Status.GPU_PASS.value,
                            whisper_text=whisper_text,
                            cer=cer,
                        )
                    else:
                        batch_rejected += 1
                        total_reject += 1
                        await self.db.update_segment(
                            seg_id,
                            status=Status.GPU_REJECT.value,
                            whisper_text=whisper_text,
                            cer=cer,
                            reject_reason=f"cer_too_high:{cer:.3f}>{threshold:.3f}",
                        )

                    processed += 1

                # Batch CER stats
                avg_cer = sum(batch_cers) / len(batch_cers) if batch_cers else 0
                min_cer = min(batch_cers) if batch_cers else 0
                max_cer = max(batch_cers) if batch_cers else 0

                logger.debug(
                    f"[Stage4] Batch {batch_num}/{total_batches}: "
                    f"pass={batch_passed}, reject={batch_rejected} "
                    f"| CER avg={avg_cer:.3f} min={min_cer:.3f} max={max_cer:.3f} "
                    f"| {batch_elapsed:.1f}s "
                    f"| Progress: {processed}/{total} ({100*processed/total:.0f}%)"
                )

                yield {
                    "item_id": f"batch_{i // batch_size}",
                    "status": "passed",
                    "batch_size": len(batch),
                    "batch_passed": batch_passed,
                    "batch_rejected": batch_rejected,
                    "processed": processed,
                    "total": total,
                }

            except (torch.cuda.OutOfMemoryError, MemoryError):
                logger.warning(
                    f"[Stage4] Batch {batch_num}/{total_batches}: OOM! "
                    f"Reducing batch_size {batch_size} → {max(1, batch_size // 2)}"
                )
                # Reduce batch size and retry
                self.batch_size = max(1, self.batch_size // 2)
                batch_size = self.batch_size
                # Requeue this batch
                await self.db.bulk_update_segments(
                    [seg["id"] for seg in batch], Status.CPU_PASS.value
                )
                yield {
                    "item_id": f"batch_{i // batch_size}",
                    "status": "retry",
                    "error": "OOM - reduced batch size",
                    "new_batch_size": batch_size,
                    "total": total,
                }

            except Exception as e:
                total_error += len(batch)
                logger.exception(
                    f"[Stage4] Batch {batch_num}/{total_batches}: ERROR - {e}"
                )
                yield {
                    "item_id": f"batch_{i // batch_size}",
                    "status": "error",
                    "error": str(e),
                    "total": total,
                }

        # Final summary
        stage_elapsed = time.time() - stage_start
        logger.info(
            f"[Stage4] === VALIDATION COMPLETE === "
            f"({stage_elapsed:.1f}s)"
        )
        logger.info(
            f"[Stage4] Results: pass={total_pass}, reject={total_reject}, "
            f"error={total_error} / total={total}"
        )
        if all_cers:
            avg_all = sum(all_cers) / len(all_cers)
            # CER distribution buckets
            buckets = {"0-0.10": 0, "0.10-0.15": 0, "0.15-0.30": 0, "0.30-0.50": 0, "0.50+": 0}
            for c in all_cers:
                if c < 0.10:
                    buckets["0-0.10"] += 1
                elif c < 0.15:
                    buckets["0.10-0.15"] += 1
                elif c < 0.30:
                    buckets["0.15-0.30"] += 1
                elif c < 0.50:
                    buckets["0.30-0.50"] += 1
                else:
                    buckets["0.50+"] += 1
            dist_str = ", ".join(f"{k}={v}" for k, v in buckets.items())
            logger.info(
                f"[Stage4] CER overall: avg={avg_all:.3f}, "
                f"min={min(all_cers):.3f}, max={max(all_cers):.3f}"
            )
            logger.info(f"[Stage4] CER distribution: {dist_str}")
