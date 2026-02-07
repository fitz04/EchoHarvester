"""Tests for database operations."""

import pytest
import pytest_asyncio

from echoharvester.db import Database, RunStatus, Status


class TestStatus:
    def test_status_values(self):
        assert Status.PENDING.value == "pending"
        assert Status.GPU_PASS.value == "gpu_pass"
        assert Status.APPROVED.value == "approved"

    def test_run_status_values(self):
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"


class TestDatabaseSource:
    @pytest.mark.asyncio
    async def test_add_source(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        assert source_id > 0

    @pytest.mark.asyncio
    async def test_add_source_upsert(self, db):
        id1 = await db.add_source("youtube_video", "https://example.com", "Test1")
        id2 = await db.add_source("youtube_video", "https://example.com", "Test2")
        assert id1 == id2  # Same type+url → upsert

    @pytest.mark.asyncio
    async def test_get_sources(self, db):
        await db.add_source("youtube_video", "https://example.com/1", "Test1")
        await db.add_source("youtube_video", "https://example.com/2", "Test2")
        sources = await db.get_sources()
        assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_get_sources_by_status(self, db):
        await db.add_source("youtube_video", "https://example.com", "Test")
        sources = await db.get_sources(status="pending")
        assert len(sources) == 1
        sources = await db.get_sources(status="done")
        assert len(sources) == 0

    @pytest.mark.asyncio
    async def test_get_source(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        source = await db.get_source(source_id)
        assert source is not None
        assert source["type"] == "youtube_video"
        assert source["label"] == "Test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_source(self, db):
        source = await db.get_source(9999)
        assert source is None

    @pytest.mark.asyncio
    async def test_update_source_status(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.update_source_status(source_id, "done")
        source = await db.get_source(source_id)
        assert source["status"] == "done"

    @pytest.mark.asyncio
    async def test_delete_source(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.delete_source(source_id)
        source = await db.get_source(source_id)
        assert source is None


class TestDatabaseMediaItem:
    @pytest.mark.asyncio
    async def test_add_media_item(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        item_id = await db.add_media_item(
            "video1", source_id, title="Test Video", duration_sec=120.0
        )
        assert item_id == "video1"

    @pytest.mark.asyncio
    async def test_get_media_items(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id, title="Video 1")
        await db.add_media_item("v2", source_id, title="Video 2")
        items = await db.get_media_items()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_get_media_items_by_status(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        items = await db.get_media_items(status="pending")
        assert len(items) == 1
        items = await db.get_media_items(status="done")
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_get_media_items_limit(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        for i in range(10):
            await db.add_media_item(f"v{i}", source_id)
        items = await db.get_media_items(limit=5)
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_get_media_item(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id, title="Test Video")
        item = await db.get_media_item("v1")
        assert item is not None
        assert item["title"] == "Test Video"

    @pytest.mark.asyncio
    async def test_update_media_item(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        await db.update_media_item("v1", status="downloaded", audio_path="/tmp/audio.wav")
        item = await db.get_media_item("v1")
        assert item["status"] == "downloaded"
        assert item["audio_path"] == "/tmp/audio.wav"

    @pytest.mark.asyncio
    async def test_update_media_item_empty(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id, title="Original")
        await db.update_media_item("v1")  # No-op
        item = await db.get_media_item("v1")
        assert item["title"] == "Original"


class TestDatabaseSegment:
    @pytest.mark.asyncio
    async def test_add_segment(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        seg_id = await db.add_segment(
            "v1_0001", "v1", 1, 1.0, 4.0, "안녕하세요", "manual"
        )
        assert seg_id == "v1_0001"

    @pytest.mark.asyncio
    async def test_add_segment_upsert_resets_on_text_change(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)

        await db.add_segment("v1_0001", "v1", 1, 1.0, 4.0, "원본 텍스트", "manual")
        await db.update_segment("v1_0001", normalized_text="정규화", cer=0.1)

        # Re-add with different original_text
        await db.add_segment("v1_0001", "v1", 1, 1.0, 4.0, "변경된 텍스트", "manual")
        segments = await db.get_segments(media_id="v1")
        seg = segments[0]
        assert seg["original_text"] == "변경된 텍스트"
        assert seg["normalized_text"] is None  # Reset
        assert seg["cer"] is None  # Reset

    @pytest.mark.asyncio
    async def test_get_segments(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        await db.add_segment("v1_0001", "v1", 0, 1.0, 4.0, "text1", "manual")
        await db.add_segment("v1_0002", "v1", 1, 5.0, 8.0, "text2", "manual")
        segments = await db.get_segments(media_id="v1")
        assert len(segments) == 2

    @pytest.mark.asyncio
    async def test_get_segments_by_status(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        await db.add_segment("v1_0001", "v1", 0, 1.0, 4.0, "text", "manual")
        await db.update_segment("v1_0001", status=Status.CPU_PASS.value)

        segments = await db.get_segments_by_status([Status.CPU_PASS.value])
        assert len(segments) == 1

    @pytest.mark.asyncio
    async def test_bulk_update_segments(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        for i in range(5):
            await db.add_segment(f"v1_{i:04d}", "v1", i, float(i), float(i + 1), f"t{i}", "manual")

        ids = [f"v1_{i:04d}" for i in range(5)]
        await db.bulk_update_segments(ids, Status.GPU_PASS.value, cer=0.05)

        segments = await db.get_segments_by_status([Status.GPU_PASS.value])
        assert len(segments) == 5
        assert all(s["cer"] == 0.05 for s in segments)

    @pytest.mark.asyncio
    async def test_bulk_update_empty(self, db):
        # Should not raise
        await db.bulk_update_segments([], Status.GPU_PASS.value)

    @pytest.mark.asyncio
    async def test_update_segment(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id)
        await db.add_segment("v1_0001", "v1", 0, 1.0, 4.0, "text", "manual")

        await db.update_segment(
            "v1_0001",
            status=Status.GPU_PASS.value,
            whisper_text="인식 텍스트",
            cer=0.05,
        )
        segments = await db.get_segments(media_id="v1")
        seg = segments[0]
        assert seg["status"] == "gpu_pass"
        assert seg["whisper_text"] == "인식 텍스트"
        assert seg["cer"] == 0.05


class TestDatabasePipelineRun:
    @pytest.mark.asyncio
    async def test_create_run(self, db):
        run_id = await db.create_run({"test": True})
        assert run_id > 0

    @pytest.mark.asyncio
    async def test_get_latest_run(self, db):
        await db.create_run({"run": 1})
        await db.create_run({"run": 2})
        latest = await db.get_latest_run()
        assert latest is not None

    @pytest.mark.asyncio
    async def test_get_latest_run_empty(self, db):
        latest = await db.get_latest_run()
        assert latest is None

    @pytest.mark.asyncio
    async def test_update_run(self, db):
        run_id = await db.create_run()
        await db.update_run(run_id, status=RunStatus.COMPLETED.value, ended=True)
        run = await db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["ended_at"] is not None

    @pytest.mark.asyncio
    async def test_get_run(self, db):
        run_id = await db.create_run({"key": "value"})
        run = await db.get_run(run_id)
        assert run is not None
        assert run["id"] == run_id


class TestDatabaseProgress:
    @pytest.mark.asyncio
    async def test_update_progress(self, db):
        run_id = await db.create_run()
        await db.update_progress(run_id, "metadata", 5, 10, "item_5")
        progress = await db.get_progress(run_id)
        assert len(progress) == 1
        assert progress[0]["processed"] == 5
        assert progress[0]["total"] == 10

    @pytest.mark.asyncio
    async def test_update_progress_upsert(self, db):
        run_id = await db.create_run()
        await db.update_progress(run_id, "metadata", 5, 10)
        await db.update_progress(run_id, "metadata", 8, 10)  # Update
        progress = await db.get_progress(run_id)
        assert len(progress) == 1
        assert progress[0]["processed"] == 8


class TestDatabaseStats:
    @pytest.mark.asyncio
    async def test_get_stats_empty(self, db):
        stats = await db.get_stats()
        assert stats["total_sources"] == 0
        assert stats["total_media_items"] == 0
        assert stats["total_segments"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id, title="Video")
        await db.add_segment("v1_0001", "v1", 0, 1.0, 5.0, "text", "manual")
        await db.update_segment(
            "v1_0001",
            status=Status.GPU_PASS.value,
            cer=0.05,
        )

        stats = await db.get_stats()
        assert stats["total_sources"] == 1
        assert stats["total_media_items"] == 1
        assert stats["total_segments"] == 1
        assert stats["segment_status"]["gpu_pass"] == 1

    @pytest.mark.asyncio
    async def test_get_recent_segments(self, db):
        source_id = await db.add_source("youtube_video", "https://example.com", "Test")
        await db.add_media_item("v1", source_id, title="Video")
        await db.add_segment("v1_0001", "v1", 0, 1.0, 5.0, "text", "manual")
        await db.update_segment("v1_0001", status=Status.GPU_PASS.value, cer=0.05)

        recent = await db.get_recent_segments(limit=10)
        assert len(recent) == 1
        assert recent[0]["media_title"] == "Video"


class TestDatabaseTransaction:
    @pytest.mark.asyncio
    async def test_transaction_commit(self, db):
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO sources (type, url_or_path, label) VALUES (?, ?, ?)",
                ("test", "path", "label"),
            )
        sources = await db.get_sources()
        assert len(sources) == 1

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db):
        with pytest.raises(ValueError):
            async with db.transaction() as conn:
                await conn.execute(
                    "INSERT INTO sources (type, url_or_path, label) VALUES (?, ?, ?)",
                    ("test", "path", "label"),
                )
                raise ValueError("rollback test")
        sources = await db.get_sources()
        assert len(sources) == 0

    @pytest.mark.asyncio
    async def test_conn_property_not_connected(self):
        db = Database(db_path="/tmp/nonexistent.db")
        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = db.conn


class TestDatabaseMigration:
    @pytest.mark.asyncio
    async def test_migrate_adds_columns(self, db):
        """Migration should add approved_at and approved_by columns."""
        cursor = await db.conn.execute("PRAGMA table_info(segments)")
        columns = [row[1] for row in await cursor.fetchall()]
        assert "approved_at" in columns
        assert "approved_by" in columns
