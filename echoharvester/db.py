"""SQLite database management for pipeline state tracking."""

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

from echoharvester.config import get_config


class Status(str, Enum):
    """Status enum for items."""

    PENDING = "pending"
    PROCESSING = "processing"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    CPU_PROCESSING = "cpu_processing"
    CPU_PASS = "cpu_pass"
    CPU_REJECT = "cpu_reject"
    GPU_PROCESSING = "gpu_processing"
    GPU_PASS = "gpu_pass"
    GPU_REJECT = "gpu_reject"
    EXPORTED = "exported"
    DONE = "done"
    ERROR = "error"


class RunStatus(str, Enum):
    """Pipeline run status."""

    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


SCHEMA = """
-- Input sources
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    url_or_path TEXT NOT NULL,
    label TEXT,
    config_json TEXT,
    status TEXT DEFAULT 'pending',
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(type, url_or_path)
);

-- Media items (videos/audio files)
CREATE TABLE IF NOT EXISTS media_items (
    id TEXT PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    title TEXT,
    duration_sec REAL,
    subtitle_type TEXT,
    source_type TEXT,
    original_path TEXT,
    audio_path TEXT,
    subtitle_path TEXT,
    status TEXT DEFAULT 'pending',
    error_msg TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Segments
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    media_id TEXT REFERENCES media_items(id) ON DELETE CASCADE,
    segment_index INTEGER,
    start_sec REAL,
    end_sec REAL,
    duration_sec REAL,
    original_text TEXT,
    normalized_text TEXT,
    whisper_text TEXT,
    cer REAL,
    snr_db REAL,
    speech_ratio REAL,
    subtitle_type TEXT,
    audio_path TEXT,
    status TEXT DEFAULT 'pending',
    reject_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline runs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT DEFAULT 'running',
    config_snapshot TEXT,
    stats_json TEXT
);

-- Progress tracking
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    stage TEXT,
    processed INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    current_item TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_media_items_status ON media_items(status);
CREATE INDEX IF NOT EXISTS idx_media_items_source ON media_items(source_id);
CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);
CREATE INDEX IF NOT EXISTS idx_segments_media ON segments(media_id);
CREATE INDEX IF NOT EXISTS idx_progress_run ON progress(run_id);
"""


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = get_config().paths.db_path
        self.db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open database connection and initialize schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._connection.row_factory = aiosqlite.Row
        await self._connection.executescript(SCHEMA)
        await self._connection.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self._connection:
            raise RuntimeError("Database not connected")
        try:
            yield self._connection
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    @property
    def conn(self) -> aiosqlite.Connection:
        """Get the database connection."""
        if not self._connection:
            raise RuntimeError("Database not connected")
        return self._connection

    # ==================== Source Methods ====================

    async def add_source(
        self,
        source_type: str,
        url_or_path: str,
        label: str = "",
        config: dict | None = None,
    ) -> int:
        """Add a new source."""
        async with self.transaction() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO sources (type, url_or_path, label, config_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(type, url_or_path) DO UPDATE SET
                    label = excluded.label,
                    config_json = excluded.config_json,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (source_type, url_or_path, label, json.dumps(config) if config else None),
            )
            row = await cursor.fetchone()
            return row["id"]

    async def get_sources(self, status: str | None = None) -> list[dict]:
        """Get all sources, optionally filtered by status."""
        if status:
            cursor = await self.conn.execute(
                "SELECT * FROM sources WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM sources ORDER BY created_at DESC"
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_source(self, source_id: int) -> dict | None:
        """Get a source by ID."""
        cursor = await self.conn.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_source_status(
        self, source_id: int, status: str, error_msg: str | None = None
    ) -> None:
        """Update source status."""
        await self.conn.execute(
            """
            UPDATE sources
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, source_id),
        )
        await self.conn.commit()

    async def delete_source(self, source_id: int) -> None:
        """Delete a source and its related data."""
        async with self.transaction() as conn:
            await conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    # ==================== Media Item Methods ====================

    async def add_media_item(
        self,
        item_id: str,
        source_id: int,
        title: str = "",
        duration_sec: float | None = None,
        subtitle_type: str | None = None,
        source_type: str = "youtube",
        original_path: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Add a new media item."""
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO media_items
                (id, source_id, title, duration_sec, subtitle_type, source_type,
                 original_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    duration_sec = excluded.duration_sec,
                    subtitle_type = excluded.subtitle_type,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    item_id,
                    source_id,
                    title,
                    duration_sec,
                    subtitle_type,
                    source_type,
                    original_path,
                    json.dumps(metadata) if metadata else None,
                ),
            )
        return item_id

    async def get_media_items(
        self,
        source_id: int | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Get media items with optional filters."""
        query = "SELECT * FROM media_items WHERE 1=1"
        params: list[Any] = []

        if source_id is not None:
            query += " AND source_id = ?"
            params.append(source_id)

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_media_item(self, item_id: str) -> dict | None:
        """Get a media item by ID."""
        cursor = await self.conn.execute(
            "SELECT * FROM media_items WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_media_item(self, item_id: str, **kwargs) -> None:
        """Update media item fields."""
        if not kwargs:
            return

        # Build SET clause
        set_parts = []
        params = []
        for key, value in kwargs.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        params.append(item_id)

        query = f"UPDATE media_items SET {', '.join(set_parts)} WHERE id = ?"
        await self.conn.execute(query, params)
        await self.conn.commit()

    # ==================== Segment Methods ====================

    async def add_segment(
        self,
        segment_id: str,
        media_id: str,
        segment_index: int,
        start_sec: float,
        end_sec: float,
        original_text: str,
        subtitle_type: str,
        audio_path: str | None = None,
    ) -> str:
        """Add a new segment."""
        duration_sec = end_sec - start_sec
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO segments
                (id, media_id, segment_index, start_sec, end_sec, duration_sec,
                 original_text, subtitle_type, audio_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    original_text = excluded.original_text,
                    audio_path = excluded.audio_path
                """,
                (
                    segment_id,
                    media_id,
                    segment_index,
                    start_sec,
                    end_sec,
                    duration_sec,
                    original_text,
                    subtitle_type,
                    audio_path,
                ),
            )
        return segment_id

    async def get_segments(
        self,
        media_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        """Get segments with optional filters."""
        query = "SELECT * FROM segments WHERE 1=1"
        params: list[Any] = []

        if media_id is not None:
            query += " AND media_id = ?"
            params.append(media_id)

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY media_id, segment_index"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_segments_by_status(
        self, statuses: list[str], limit: int = 100
    ) -> list[dict]:
        """Get segments by multiple statuses."""
        placeholders = ",".join(["?"] * len(statuses))
        cursor = await self.conn.execute(
            f"""
            SELECT * FROM segments
            WHERE status IN ({placeholders})
            ORDER BY created_at
            LIMIT ?
            """,
            (*statuses, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def update_segment(self, segment_id: str, **kwargs) -> None:
        """Update segment fields."""
        if not kwargs:
            return

        set_parts = []
        params = []
        for key, value in kwargs.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        params.append(segment_id)
        query = f"UPDATE segments SET {', '.join(set_parts)} WHERE id = ?"
        await self.conn.execute(query, params)
        await self.conn.commit()

    async def bulk_update_segments(
        self, segment_ids: list[str], status: str, **kwargs
    ) -> None:
        """Bulk update segment status."""
        if not segment_ids:
            return

        placeholders = ",".join(["?"] * len(segment_ids))
        set_parts = ["status = ?"]
        params: list[Any] = [status]

        for key, value in kwargs.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        params.extend(segment_ids)
        query = f"""
            UPDATE segments
            SET {', '.join(set_parts)}
            WHERE id IN ({placeholders})
        """
        await self.conn.execute(query, params)
        await self.conn.commit()

    # ==================== Pipeline Run Methods ====================

    async def create_run(self, config_snapshot: dict | None = None) -> int:
        """Create a new pipeline run."""
        async with self.transaction() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO pipeline_runs (started_at, config_snapshot)
                VALUES (?, ?)
                RETURNING id
                """,
                (datetime.now(), json.dumps(config_snapshot) if config_snapshot else None),
            )
            row = await cursor.fetchone()
            return row["id"]

    async def update_run(
        self,
        run_id: int,
        status: str | None = None,
        stats: dict | None = None,
        ended: bool = False,
    ) -> None:
        """Update pipeline run."""
        set_parts = []
        params: list[Any] = []

        if status:
            set_parts.append("status = ?")
            params.append(status)

        if stats:
            set_parts.append("stats_json = ?")
            params.append(json.dumps(stats))

        if ended:
            set_parts.append("ended_at = ?")
            params.append(datetime.now())

        if not set_parts:
            return

        params.append(run_id)
        query = f"UPDATE pipeline_runs SET {', '.join(set_parts)} WHERE id = ?"
        await self.conn.execute(query, params)
        await self.conn.commit()

    async def get_latest_run(self) -> dict | None:
        """Get the latest pipeline run."""
        cursor = await self.conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_run(self, run_id: int) -> dict | None:
        """Get a pipeline run by ID."""
        cursor = await self.conn.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ==================== Progress Methods ====================

    async def update_progress(
        self,
        run_id: int,
        stage: str,
        processed: int,
        total: int,
        current_item: str = "",
    ) -> None:
        """Update or insert progress for a stage."""
        await self.conn.execute(
            """
            INSERT INTO progress (run_id, stage, processed, total, current_item, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, stage) DO UPDATE SET
                processed = excluded.processed,
                total = excluded.total,
                current_item = excluded.current_item,
                updated_at = CURRENT_TIMESTAMP
            """,
            (run_id, stage, processed, total, current_item),
        )
        await self.conn.commit()

    async def get_progress(self, run_id: int) -> list[dict]:
        """Get progress for all stages of a run."""
        cursor = await self.conn.execute(
            "SELECT * FROM progress WHERE run_id = ? ORDER BY stage",
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ==================== Statistics Methods ====================

    async def get_stats(self) -> dict:
        """Get overall pipeline statistics."""
        stats = {}

        # Total counts
        cursor = await self.conn.execute("SELECT COUNT(*) as count FROM sources")
        stats["total_sources"] = (await cursor.fetchone())["count"]

        cursor = await self.conn.execute("SELECT COUNT(*) as count FROM media_items")
        stats["total_media_items"] = (await cursor.fetchone())["count"]

        cursor = await self.conn.execute("SELECT COUNT(*) as count FROM segments")
        stats["total_segments"] = (await cursor.fetchone())["count"]

        # Status breakdown for segments
        cursor = await self.conn.execute(
            """
            SELECT status, COUNT(*) as count
            FROM segments
            GROUP BY status
            """
        )
        stats["segment_status"] = {row["status"]: row["count"] for row in await cursor.fetchall()}

        # Duration statistics
        cursor = await self.conn.execute(
            """
            SELECT
                SUM(duration_sec) as total_duration,
                AVG(cer) as avg_cer
            FROM segments
            WHERE status IN ('gpu_pass', 'exported')
            """
        )
        row = await cursor.fetchone()
        stats["passed_duration_hours"] = (row["total_duration"] or 0) / 3600
        stats["avg_cer"] = row["avg_cer"]

        # CER distribution
        cursor = await self.conn.execute(
            """
            SELECT
                CASE
                    WHEN cer < 0.05 THEN '0-5%'
                    WHEN cer < 0.10 THEN '5-10%'
                    WHEN cer < 0.15 THEN '10-15%'
                    ELSE '15%+'
                END as cer_range,
                COUNT(*) as count
            FROM segments
            WHERE cer IS NOT NULL AND status IN ('gpu_pass', 'exported')
            GROUP BY cer_range
            ORDER BY cer_range
            """
        )
        stats["cer_distribution"] = {
            row["cer_range"]: row["count"] for row in await cursor.fetchall()
        }

        return stats

    async def get_recent_segments(self, limit: int = 20) -> list[dict]:
        """Get recently processed segments."""
        cursor = await self.conn.execute(
            """
            SELECT s.*, m.title as media_title
            FROM segments s
            JOIN media_items m ON s.media_id = m.id
            WHERE s.status IN ('gpu_pass', 'gpu_reject', 'exported')
            ORDER BY s.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# Global database instance
_db: Database | None = None


async def get_db() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
        await _db.connect()
    return _db


async def close_db() -> None:
    """Close the global database instance."""
    global _db
    if _db:
        await _db.close()
        _db = None
