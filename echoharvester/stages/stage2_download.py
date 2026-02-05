"""Stage 2: Download/Prepare - Download YouTube videos or prepare local files."""

import asyncio
import logging
from typing import Any, AsyncGenerator

from echoharvester.config import Config
from echoharvester.db import Database, Status
from echoharvester.sources import BaseSource, SourceType
from echoharvester.stages.base import BaseStage

logger = logging.getLogger(__name__)


class DownloadStage(BaseStage):
    """Download/prepare stage."""

    name = "download"

    def __init__(self, config: Config, db: Database):
        super().__init__(config, db)
        self.semaphore = asyncio.Semaphore(config.download.max_concurrent)

    async def _download_item(self, item: dict, source: BaseSource) -> dict:
        """Download a single item with semaphore control."""
        async with self.semaphore:
            item_id = item["id"]

            try:
                # Update status
                await self.db.update_media_item(
                    item_id, status=Status.DOWNLOADING.value
                )

                # Create MediaItem from database record
                from echoharvester.sources.base import MediaItem, SourceType, SubtitleType
                from pathlib import Path

                media_item = MediaItem(
                    id=item_id,
                    source_id=item["source_id"],
                    title=item.get("title", ""),
                    duration_sec=item.get("duration_sec"),
                    subtitle_type=SubtitleType(item.get("subtitle_type", "none")),
                    source_type=SourceType(item.get("source_type", "local_file")),
                    url=item.get("original_path") if "youtube" in item.get("source_type", "") else None,
                    original_path=Path(item["original_path"]) if item.get("original_path") else None,
                )

                # Prepare (download or convert)
                prepared = await source.prepare(
                    media_item, self.config.paths.work_dir
                )

                # Update database with paths
                await self.db.update_media_item(
                    item_id,
                    status=Status.DOWNLOADED.value,
                    audio_path=str(prepared.audio_path),
                    subtitle_path=str(prepared.subtitle_path) if prepared.subtitle_path else None,
                )

                return {
                    "item_id": item_id,
                    "status": "passed",
                    "audio_path": str(prepared.audio_path),
                    "subtitle_path": str(prepared.subtitle_path) if prepared.subtitle_path else None,
                }

            except Exception as e:
                logger.error(f"Error downloading {item_id}: {e}")
                await self.db.update_media_item(
                    item_id,
                    status=Status.ERROR.value,
                    error_msg=str(e),
                )
                return {
                    "item_id": item_id,
                    "status": "error",
                    "error": str(e),
                }

    async def process(self) -> AsyncGenerator[dict[str, Any], None]:
        """Download/prepare pending media items."""
        # Get pending items
        items = await self.db.get_media_items(status=Status.PENDING.value)
        total = len(items)

        if total == 0:
            logger.info("No items to download")
            return

        logger.info(f"Downloading/preparing {total} items")

        # Group items by source
        items_by_source: dict[int, list[dict]] = {}
        for item in items:
            source_id = item["source_id"]
            if source_id not in items_by_source:
                items_by_source[source_id] = []
            items_by_source[source_id].append(item)

        # Create sources
        sources: dict[int, BaseSource] = {}
        for source_id in items_by_source.keys():
            source_data = await self.db.get_source(source_id)
            if source_data:
                source_config = {
                    "type": source_data["type"],
                    "url": source_data.get("url_or_path"),
                    "path": source_data.get("url_or_path"),
                    "label": source_data.get("label", ""),
                }
                sources[source_id] = BaseSource.from_config(source_id, source_config)

        # Process items
        processed = 0
        passed = 0
        failed = 0

        # Create download tasks
        tasks = []
        for item in items:
            if self._cancelled:
                break

            source_id = item["source_id"]
            source = sources.get(source_id)

            if not source:
                logger.error(f"Source not found for item {item['id']}")
                continue

            task = self._download_item(item, source)
            tasks.append(task)

        # Process with limited concurrency
        for coro in asyncio.as_completed(tasks):
            if self._cancelled:
                break

            result = await coro
            processed += 1

            if result["status"] == "passed":
                passed += 1
            else:
                failed += 1

            yield {
                **result,
                "processed": processed,
                "total": total,
            }

        logger.info(f"Download complete: {passed} passed, {failed} failed")
