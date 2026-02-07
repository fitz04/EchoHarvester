"""Stage 1: Metadata Collection - Discover media items from sources."""

import logging
from typing import Any, AsyncGenerator

from echoharvester.config import Config
from echoharvester.db import Database, Status
from echoharvester.sources import BaseSource
from echoharvester.stages.base import BaseStage

logger = logging.getLogger(__name__)


class MetadataStage(BaseStage):
    """Metadata collection stage."""

    name = "metadata"

    def __init__(self, config: Config, db: Database):
        super().__init__(config, db)

    async def process(self) -> AsyncGenerator[dict[str, Any], None]:
        """Discover media items from all configured sources."""
        # Get all sources
        sources = await self.db.get_sources()
        total_sources = len(sources)

        if total_sources == 0:
            # Add sources from config
            for source_config in self.config.sources:
                source_id = await self.db.add_source(
                    source_type=source_config.type,
                    url_or_path=source_config.url or source_config.path or "",
                    label=source_config.label,
                    config=source_config.model_dump(),
                )
                logger.info(f"Added source {source_id}: {source_config.label}")

            sources = await self.db.get_sources()
            total_sources = len(sources)

        logger.info(f"Discovering media from {total_sources} sources")

        total_items = 0

        for idx, source_data in enumerate(sources):
            if self._cancelled:
                break

            source_id = source_data["id"]
            source_type = source_data["type"]

            try:
                # Update source status
                await self.db.update_source_status(source_id, Status.PROCESSING.value)

                # Create source instance
                source_config = {
                    "type": source_type,
                    "url": source_data.get("url_or_path"),
                    "path": source_data.get("url_or_path"),
                    "label": source_data.get("label", ""),
                }

                source = BaseSource.from_config(source_id, source_config)

                # Discover items
                items = await source.discover()
                logger.info(f"Discovered {len(items)} items from source {source_id}")

                # Save to database
                for item in items:
                    # Skip items without subtitles if configured
                    if item.subtitle_type.value == "none":
                        logger.debug(f"Skipping {item.id}: no subtitle")
                        continue

                    await self.db.add_media_item(
                        item_id=item.id,
                        source_id=source_id,
                        title=item.title,
                        duration_sec=item.duration_sec,
                        subtitle_type=item.subtitle_type.value,
                        source_type=item.source_type.value,
                        original_path=str(item.original_path) if item.original_path else None,
                        metadata=item.metadata,
                    )
                    total_items += 1

                # Update source status
                await self.db.update_source_status(source_id, Status.DONE.value)

                yield {
                    "item_id": f"source_{source_id}",
                    "status": "passed",
                    "items_discovered": len(items),
                    "total_items": total_items,
                    "total": total_sources,
                }

            except Exception as e:
                logger.exception(f"Error processing source {source_id}: {e}")
                await self.db.update_source_status(
                    source_id, Status.ERROR.value, str(e)
                )
                yield {
                    "item_id": f"source_{source_id}",
                    "status": "error",
                    "error": str(e),
                    "total": total_sources,
                }

        logger.info(f"Metadata collection complete: {total_items} items")
