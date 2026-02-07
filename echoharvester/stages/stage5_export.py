"""Stage 5: Lhotse Shar Export - Package validated segments into Lhotse format."""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from echoharvester.config import Config
from echoharvester.db import Database, Status
from echoharvester.stages.base import BaseStage

logger = logging.getLogger(__name__)


class ExportStage(BaseStage):
    """Export validated segments to Lhotse Shar format."""

    name = "export"

    def __init__(self, config: Config, db: Database):
        super().__init__(config, db)
        self.output_dir = config.paths.output_dir
        self.shar_dir = self.output_dir / "shar"

    async def process(self) -> AsyncGenerator[dict[str, Any], None]:
        """Export GPU-passed segments to Lhotse format."""
        from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
        from lhotse.shar import SharWriter

        # Get all validated segments
        segments = await self.db.get_segments_by_status(
            [Status.GPU_PASS.value, Status.APPROVED.value],
            limit=100000,
        )
        total = len(segments)

        if total == 0:
            logger.info("No segments to export")
            return

        logger.info(f"Exporting {total} segments to Lhotse Shar")

        # Create output directory
        self.shar_dir.mkdir(parents=True, exist_ok=True)

        # Build CutSet
        cuts = []
        processed = 0
        errors = 0

        for seg in segments:
            if self._cancelled:
                break

            try:
                seg_id = seg["id"]
                audio_path = seg.get("audio_path")

                if not audio_path or not Path(audio_path).exists():
                    logger.warning(f"Audio not found for segment {seg_id}")
                    errors += 1
                    continue

                # Create Recording
                recording = Recording.from_file(audio_path, recording_id=seg_id)

                # Create Supervision
                supervision = SupervisionSegment(
                    id=seg_id,
                    recording_id=seg_id,
                    start=0.0,
                    duration=recording.duration,
                    text=seg.get("normalized_text", ""),
                    language="ko",
                )

                # Create Cut with custom fields
                cut = MonoCut(
                    id=seg_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    recording=recording,
                    supervisions=[supervision],
                    custom={
                        "source_type": seg.get("subtitle_type", "unknown"),
                        "cer": seg.get("cer"),
                        "snr_db": seg.get("snr_db"),
                        "speech_ratio": seg.get("speech_ratio"),
                        "media_id": seg.get("media_id"),
                        "original_text": seg.get("original_text"),
                        "whisper_text": seg.get("whisper_text"),
                    },
                )

                cuts.append(cut)
                processed += 1

                # Update segment status
                await self.db.update_segment(seg_id, status=Status.EXPORTED.value)

                # Yield progress every 100 segments
                if processed % 100 == 0:
                    yield {
                        "item_id": seg_id,
                        "status": "passed",
                        "processed": processed,
                        "total": total,
                    }

            except Exception as e:
                logger.error(f"Error processing segment {seg.get('id')}: {e}")
                errors += 1

        if not cuts:
            logger.warning("No cuts to export")
            yield {
                "item_id": "export",
                "status": "failed",
                "error": "No valid segments",
                "total": total,
            }
            return

        # Create CutSet
        cutset = CutSet.from_cuts(cuts)

        # Export to Shar format
        logger.info(f"Writing {len(cutset)} cuts to Shar format")

        try:
            with SharWriter(
                str(self.shar_dir),
                fields={"recording": "wav"},
                shard_size=1000,  # cuts per shard
            ) as writer:
                for cut in cutset:
                    writer.write(cut)

            # Also save as regular manifest for convenience
            manifest_path = self.output_dir / "manifest.jsonl.gz"
            cutset.to_jsonl(manifest_path)

            # Generate statistics report
            stats = await self._generate_stats(cutset)

            yield {
                "item_id": "export",
                "status": "passed",
                "processed": processed,
                "errors": errors,
                "total": total,
                "stats": stats,
            }

        except Exception as e:
            logger.exception(f"Error writing Shar: {e}")
            yield {
                "item_id": "export",
                "status": "error",
                "error": str(e),
                "processed": processed,
                "total": total,
            }

    async def _generate_stats(self, cutset) -> dict:
        """Generate export statistics."""
        total_duration = sum(cut.duration for cut in cutset)
        cer_values = [cut.custom.get("cer", 0) for cut in cutset if cut.custom.get("cer")]
        snr_values = [cut.custom.get("snr_db", 0) for cut in cutset if cut.custom.get("snr_db")]

        stats = {
            "total_cuts": len(cutset),
            "total_duration_sec": total_duration,
            "total_duration_hours": total_duration / 3600,
            "avg_duration_sec": total_duration / len(cutset) if cutset else 0,
            "avg_cer": sum(cer_values) / len(cer_values) if cer_values else 0,
            "avg_snr_db": sum(snr_values) / len(snr_values) if snr_values else 0,
            "cer_distribution": {},
            "subtitle_type_distribution": {},
        }

        # CER distribution
        for cut in cutset:
            cer = cut.custom.get("cer", 0)
            if cer < 0.05:
                bucket = "0-5%"
            elif cer < 0.10:
                bucket = "5-10%"
            elif cer < 0.15:
                bucket = "10-15%"
            else:
                bucket = "15%+"
            stats["cer_distribution"][bucket] = stats["cer_distribution"].get(bucket, 0) + 1

        # Subtitle type distribution
        for cut in cutset:
            sub_type = cut.custom.get("source_type", "unknown")
            stats["subtitle_type_distribution"][sub_type] = (
                stats["subtitle_type_distribution"].get(sub_type, 0) + 1
            )

        # Save stats to file
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Export stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")

        return stats
