"""Stage 3: CPU Preprocessing - Subtitle parsing, normalization, segmentation, filtering."""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, AsyncGenerator

from echoharvester.config import Config, get_config
from echoharvester.db import Database, Status
from echoharvester.stages.base import BaseStage
from echoharvester.utils import (
    ForcedAligner,
    SubtitleSegment,
    align_subtitles_with_audio,
    calculate_snr_from_file,
    extract_segment_sync,
    is_valid_text,
    merge_overlapping_segments,
    normalize_text,
    parse_subtitle,
    split_long_segments,
)
from echoharvester.utils.vad import calculate_speech_ratio_from_file

logger = logging.getLogger(__name__)


def process_segment_cpu(
    segment_data: dict,
    audio_path: str,
    output_dir: str,
    config_dict: dict,
) -> dict:
    """CPU processing for a single segment (runs in process pool).

    Args:
        segment_data: Segment information
        audio_path: Path to source audio
        output_dir: Output directory for segment audio
        config_dict: Configuration as dictionary

    Returns:
        Processing result dictionary
    """
    try:
        segment_id = segment_data["id"]
        start_sec = segment_data["start_sec"]
        end_sec = segment_data["end_sec"]
        original_text = segment_data["original_text"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize text
        normalized_text = normalize_text(original_text)

        # Check text validity
        if not is_valid_text(normalized_text):
            return {
                "segment_id": segment_id,
                "status": "rejected",
                "reason": "invalid_text",
            }

        duration = end_sec - start_sec

        # Check duration
        if duration < config_dict["min_duration_sec"]:
            return {
                "segment_id": segment_id,
                "status": "rejected",
                "reason": "too_short",
            }

        if duration > config_dict["max_duration_sec"]:
            return {
                "segment_id": segment_id,
                "status": "rejected",
                "reason": "too_long",
            }

        # Extract audio segment
        segment_audio_path = output_dir / f"{segment_id}.wav"
        extract_segment_sync(
            audio_path,
            segment_audio_path,
            start_sec,
            end_sec,
            padding_sec=config_dict.get("segment_padding_sec", 0.15),
            sample_rate=config_dict.get("sample_rate", 16000),
        )

        # Calculate SNR
        snr_db = calculate_snr_from_file(segment_audio_path)
        if snr_db < config_dict["min_snr_db"]:
            return {
                "segment_id": segment_id,
                "status": "rejected",
                "reason": "low_snr",
                "snr_db": snr_db,
            }

        # Calculate speech ratio using VAD
        speech_ratio = calculate_speech_ratio_from_file(segment_audio_path)
        if speech_ratio < config_dict["min_speech_ratio"]:
            return {
                "segment_id": segment_id,
                "status": "rejected",
                "reason": "low_speech_ratio",
                "speech_ratio": speech_ratio,
            }

        return {
            "segment_id": segment_id,
            "status": "passed",
            "normalized_text": normalized_text,
            "audio_path": str(segment_audio_path),
            "snr_db": snr_db,
            "speech_ratio": speech_ratio,
        }

    except Exception as e:
        return {
            "segment_id": segment_data.get("id", "unknown"),
            "status": "error",
            "error": str(e),
        }


class PreprocessStage(BaseStage):
    """CPU preprocessing stage."""

    name = "preprocess"

    def __init__(self, config: Config, db: Database):
        super().__init__(config, db)
        self.executor = ProcessPoolExecutor(
            max_workers=config.pipeline.num_cpu_workers
        )
        self._aligner = None

    def _init_aligner(self):
        """Initialize forced aligner if enabled."""
        if self._aligner is not None:
            return
        fa_config = self.config.forced_alignment
        if not fa_config.enabled:
            return
        self._aligner = ForcedAligner(
            model_name=fa_config.model,
            device=fa_config.device,
            compute_type=fa_config.compute_type,
            language=fa_config.language,
        )

    async def process(self) -> AsyncGenerator[dict[str, Any], None]:
        """Process downloaded media items."""
        # Get media items ready for preprocessing
        items = await self.db.get_media_items(status=Status.DOWNLOADED.value)
        total = len(items)

        logger.info(f"Preprocessing {total} media items")

        for idx, item in enumerate(items):
            if self._cancelled:
                break

            item_id = item["id"]
            audio_path = item.get("audio_path")
            subtitle_path = item.get("subtitle_path")

            if not audio_path or not Path(audio_path).exists():
                yield {
                    "item_id": item_id,
                    "status": "error",
                    "error": "Audio file not found",
                    "total": total,
                }
                continue

            if not subtitle_path or not Path(subtitle_path).exists():
                yield {
                    "item_id": item_id,
                    "status": "failed",
                    "error": "Subtitle file not found",
                    "total": total,
                }
                await self.db.update_media_item(
                    item_id,
                    status=Status.ERROR.value,
                    error_msg="No subtitle file",
                )
                continue

            try:
                # Update status
                await self.db.update_media_item(
                    item_id, status=Status.CPU_PROCESSING.value
                )

                # Parse subtitles
                segments = parse_subtitle(subtitle_path)

                # Merge overlapping segments
                segments = merge_overlapping_segments(segments)

                # Split long segments
                segments = split_long_segments(
                    segments,
                    max_duration_sec=self.config.filters.max_duration_sec,
                )

                # Forced alignment: refine timestamps using Qwen3-ForcedAligner
                if self.config.forced_alignment.enabled:
                    self._init_aligner()
                    if self._aligner is not None:
                        logger.info(
                            f"Running forced alignment on {len(segments)} segments"
                        )
                        segments = align_subtitles_with_audio(
                            audio_path, segments, self._aligner,
                        )
                        # Free FA model memory before Stage 4 loads ASR
                        self._aligner.unload()
                        self._aligner = None

                # Prepare segment data
                segment_dir = (
                    self.config.paths.work_dir / "segments" / item_id
                )
                subtitle_type = item.get("subtitle_type", "unknown")

                segment_data_list = []
                for i, seg in enumerate(segments):
                    seg_id = f"{item_id}_{i:04d}"

                    # Add to database
                    await self.db.add_segment(
                        segment_id=seg_id,
                        media_id=item_id,
                        segment_index=i,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        original_text=seg.text,
                        subtitle_type=subtitle_type,
                    )

                    segment_data_list.append({
                        "id": seg_id,
                        "start_sec": seg.start_sec,
                        "end_sec": seg.end_sec,
                        "original_text": seg.text,
                    })

                # Process segments in parallel using process pool
                config_dict = {
                    "min_duration_sec": self.config.filters.min_duration_sec,
                    "max_duration_sec": self.config.filters.max_duration_sec,
                    "min_snr_db": self.config.filters.min_snr_db,
                    "min_speech_ratio": self.config.filters.min_speech_ratio,
                    "segment_padding_sec": self.config.audio.segment_padding_sec,
                    "sample_rate": self.config.audio.sample_rate,
                }

                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        self.executor,
                        process_segment_cpu,
                        seg_data,
                        audio_path,
                        str(segment_dir),
                        config_dict,
                    )
                    for seg_data in segment_data_list
                ]

                passed_count = 0
                rejected_count = 0
                error_count = 0
                total_segs = len(segment_data_list)
                reject_reasons: dict[str, int] = {}

                logger.info(
                    f"[Stage3] Processing {total_segs} segments "
                    f"(CPU workers={self.config.pipeline.num_cpu_workers})"
                )

                for result in await asyncio.gather(*tasks, return_exceptions=True):
                    if isinstance(result, Exception):
                        error_count += 1
                        logger.error(f"Segment processing error: {result}")
                        continue

                    seg_id = result.get("segment_id")
                    if result["status"] == "passed":
                        passed_count += 1
                        await self.db.update_segment(
                            seg_id,
                            status=Status.CPU_PASS.value,
                            normalized_text=result.get("normalized_text"),
                            audio_path=result.get("audio_path"),
                            snr_db=result.get("snr_db"),
                            speech_ratio=result.get("speech_ratio"),
                        )
                    elif result["status"] == "rejected":
                        rejected_count += 1
                        reason = result.get("reason", "unknown")
                        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                        await self.db.update_segment(
                            seg_id,
                            status=Status.CPU_REJECT.value,
                            reject_reason=reason,
                            snr_db=result.get("snr_db"),
                            speech_ratio=result.get("speech_ratio"),
                        )
                    else:
                        error_count += 1
                        await self.db.update_segment(
                            seg_id,
                            status=Status.ERROR.value,
                            reject_reason=result.get("error"),
                        )

                # Log summary
                logger.info(
                    f"[Stage3] {item_id}: "
                    f"pass={passed_count}, reject={rejected_count}, "
                    f"error={error_count} / total={total_segs}"
                )
                if reject_reasons:
                    reasons_str = ", ".join(
                        f"{k}={v}" for k, v in sorted(
                            reject_reasons.items(), key=lambda x: -x[1]
                        )
                    )
                    logger.info(f"[Stage3] Reject breakdown: {reasons_str}")

                # Update media item status
                await self.db.update_media_item(
                    item_id,
                    status=Status.PROCESSING.value,
                )

                yield {
                    "item_id": item_id,
                    "status": "passed" if passed_count > 0 else "failed",
                    "total_segments": len(segment_data_list),
                    "passed_segments": passed_count,
                    "rejected_segments": rejected_count,
                    "total": total,
                }

            except Exception as e:
                logger.error(
                    f"[Stage3] Error preprocessing {item_id}: "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                await self.db.update_media_item(
                    item_id,
                    status=Status.ERROR.value,
                    error_msg=f"{type(e).__name__}: {e}",
                )
                yield {
                    "item_id": item_id,
                    "status": "error",
                    "error": str(e),
                    "total": total,
                }

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
