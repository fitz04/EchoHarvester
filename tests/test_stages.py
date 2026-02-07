"""Tests for pipeline stages."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echoharvester.db import Status
from echoharvester.stages.base import BaseStage


class TestBaseStage:
    def test_cancel(self, sample_config, db):
        """Test that cancel sets the flag."""

        class ConcreteStage(BaseStage):
            name = "test"

            async def process(self):
                yield {"status": "passed", "total": 1}

        import asyncio
        loop = asyncio.new_event_loop()
        database = loop.run_until_complete(self._make_db(sample_config))
        stage = ConcreteStage(sample_config, database)
        assert stage._cancelled is False
        stage.cancel()
        assert stage._cancelled is True
        loop.run_until_complete(database.close())
        loop.close()

    async def _make_db(self, config):
        from echoharvester.db import Database
        db = Database(db_path=config.paths.db_path)
        await db.connect()
        return db

    @pytest.mark.asyncio
    async def test_run_tracks_progress(self, sample_config, db):
        """Test that run() tracks processed/passed/failed counts."""

        class TestStage(BaseStage):
            name = "test"

            async def process(self):
                yield {"status": "passed", "item_id": "1", "total": 3}
                yield {"status": "passed", "item_id": "2", "total": 3}
                yield {"status": "failed", "item_id": "3", "total": 3}

        run_id = await db.create_run()
        stage = TestStage(sample_config, db)
        stats = await stage.run(run_id)
        assert stats["processed"] == 3
        assert stats["passed"] == 2
        assert stats["failed"] == 1
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_run_handles_errors(self, sample_config, db):
        """Test that run() counts errors."""

        class ErrorStage(BaseStage):
            name = "test"

            async def process(self):
                yield {"status": "error", "error": "something", "total": 1}

        run_id = await db.create_run()
        stage = ErrorStage(sample_config, db)
        stats = await stage.run(run_id)
        assert stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_run_exception_propagates(self, sample_config, db):
        """Test that exceptions in process() propagate."""

        class BrokenStage(BaseStage):
            name = "test"

            async def process(self):
                raise RuntimeError("stage broken")
                yield  # noqa: unreachable

        run_id = await db.create_run()
        stage = BrokenStage(sample_config, db)
        with pytest.raises(RuntimeError, match="stage broken"):
            await stage.run(run_id)
        assert stage.is_running is False

    @pytest.mark.asyncio
    async def test_is_running(self, sample_config, db):
        class QuickStage(BaseStage):
            name = "test"

            async def process(self):
                yield {"status": "passed", "total": 1}

        run_id = await db.create_run()
        stage = QuickStage(sample_config, db)
        assert stage.is_running is False
        await stage.run(run_id)
        assert stage.is_running is False


class TestMetadataStage:
    @pytest.mark.asyncio
    async def test_no_sources(self, sample_config, db):
        """Metadata stage with no sources should process nothing."""
        from echoharvester.stages.stage1_metadata import MetadataStage

        stage = MetadataStage(sample_config, db)
        run_id = await db.create_run()
        stats = await stage.run(run_id)
        assert stats["processed"] == 0


class TestExportStage:
    @pytest.mark.asyncio
    async def test_no_segments_to_export(self, sample_config, db):
        """Export stage with no passed segments should handle gracefully."""
        pytest.importorskip("lhotse")
        from echoharvester.stages.stage5_export import ExportStage

        stage = ExportStage(sample_config, db)
        run_id = await db.create_run()
        stats = await stage.run(run_id)
        assert stats["processed"] == 0
