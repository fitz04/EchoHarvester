"""Stage 4: GPU Validation - Whisper re-transcription and CER filtering."""

import logging
from pathlib import Path
from typing import Any, AsyncGenerator

import torch

from echoharvester.config import Config
from echoharvester.db import Database, Status
from echoharvester.stages.base import BaseStage
from echoharvester.utils import calculate_cer, is_acceptable_cer, normalize_for_cer

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Wrapper for faster-whisper transcription."""

    def __init__(
        self,
        model_name: str = "seastar105/whisper-medium-komixv2",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ko",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {self.model_name}")

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        logger.info("Whisper model loaded")

    def transcribe(self, audio_path: str | Path, beam_size: int = 5) -> str:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to audio file
            beam_size: Beam search size

        Returns:
            Transcribed text
        """
        self._load_model()

        segments, _ = self._model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            language=self.language,
            vad_filter=False,  # We already did VAD
        )

        text = " ".join(segment.text for segment in segments)
        return text.strip()

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
        beam_size: int = 5,
    ) -> list[str]:
        """Transcribe multiple audio files.

        Note: faster-whisper doesn't have native batching,
        so we process sequentially but keep model loaded.
        """
        self._load_model()

        results = []
        for audio_path in audio_paths:
            try:
                text = self.transcribe(audio_path, beam_size)
                results.append(text)
            except Exception as e:
                logger.error(f"Transcription error for {audio_path}: {e}")
                results.append("")

        return results


class ValidateStage(BaseStage):
    """GPU validation stage using Whisper."""

    name = "validate"

    def __init__(self, config: Config, db: Database):
        super().__init__(config, db)
        self.transcriber = None
        self.batch_size = config.validation.batch_size

    def _init_transcriber(self):
        """Initialize the transcriber."""
        if self.transcriber is not None:
            return

        self.transcriber = WhisperTranscriber(
            model_name=self.config.validation.model,
            device=self.config.validation.device,
            compute_type=self.config.validation.compute_type,
            language=self.config.validation.language,
        )

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

        logger.info(f"Validating {total} segments")
        self._init_transcriber()

        # Process in batches
        batch_size = self.batch_size
        processed = 0

        for i in range(0, total, batch_size):
            if self._cancelled:
                break

            batch = segments[i : i + batch_size]
            audio_paths = [seg["audio_path"] for seg in batch]

            try:
                # Mark as processing
                segment_ids = [seg["id"] for seg in batch]
                await self.db.bulk_update_segments(
                    segment_ids, Status.GPU_PROCESSING.value
                )

                # Transcribe batch
                transcriptions = self.transcriber.transcribe_batch(
                    audio_paths,
                    beam_size=self.config.validation.beam_size,
                )

                # Calculate CER and update status
                for seg, whisper_text in zip(batch, transcriptions):
                    seg_id = seg["id"]
                    normalized_text = seg.get("normalized_text", "")
                    subtitle_type = seg.get("subtitle_type", "unknown")

                    # Calculate CER
                    cer = calculate_cer(normalized_text, whisper_text)

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
                        await self.db.update_segment(
                            seg_id,
                            status=Status.GPU_PASS.value,
                            whisper_text=whisper_text,
                            cer=cer,
                        )
                    else:
                        await self.db.update_segment(
                            seg_id,
                            status=Status.GPU_REJECT.value,
                            whisper_text=whisper_text,
                            cer=cer,
                            reject_reason=f"cer_too_high:{cer:.3f}>{threshold:.3f}",
                        )

                    processed += 1

                # Calculate batch stats
                batch_passed = sum(
                    1
                    for seg, text in zip(batch, transcriptions)
                    if is_acceptable_cer(
                        calculate_cer(seg.get("normalized_text", ""), text),
                        seg.get("subtitle_type", "unknown"),
                        self.config.filters.cer_threshold_manual,
                        self.config.filters.cer_threshold_auto,
                    )
                )

                yield {
                    "item_id": f"batch_{i // batch_size}",
                    "status": "passed",
                    "batch_size": len(batch),
                    "batch_passed": batch_passed,
                    "batch_rejected": len(batch) - batch_passed,
                    "processed": processed,
                    "total": total,
                }

            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU OOM, reducing batch size")
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
                    "error": "GPU OOM - reduced batch size",
                    "new_batch_size": batch_size,
                    "total": total,
                }

            except Exception as e:
                logger.exception(f"Batch validation error: {e}")
                yield {
                    "item_id": f"batch_{i // batch_size}",
                    "status": "error",
                    "error": str(e),
                    "total": total,
                }
