"""Tests for pipeline orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echoharvester.pipeline.orchestrator import PipelineOrchestrator, PipelineState


class TestPipelineState:
    def test_values(self):
        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.RUNNING.value == "running"
        assert PipelineState.COMPLETED.value == "completed"
        assert PipelineState.ERROR.value == "error"


class TestPipelineOrchestrator:
    def test_initial_state(self):
        orch = PipelineOrchestrator()
        assert orch.state == PipelineState.IDLE
        assert orch.current_stage is None
        assert orch.run_id is None

    def test_stages_list(self):
        assert PipelineOrchestrator.STAGES == [
            "metadata", "download", "preprocess", "validate", "export"
        ]

    @pytest.mark.asyncio
    async def test_initialize(self, sample_config, db):
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()
        assert "metadata" in orch._stages
        assert "download" in orch._stages
        assert "preprocess" in orch._stages
        assert "validate" in orch._stages
        assert "export" in orch._stages

    @pytest.mark.asyncio
    async def test_get_status(self, sample_config, db):
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()
        status = await orch.get_status()
        assert status["state"] == "idle"
        assert status["current_stage"] is None

    @pytest.mark.asyncio
    async def test_get_stats(self, sample_config, db):
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()
        stats = await orch.get_stats()
        assert "total_sources" in stats
        assert "total_segments" in stats

    def test_stop_when_idle(self, sample_config):
        orch = PipelineOrchestrator(config=sample_config)
        orch.stop()  # Should not raise
        assert orch.state == PipelineState.IDLE

    def test_set_progress_callback(self, sample_config):
        orch = PipelineOrchestrator(config=sample_config)
        callback = MagicMock()
        orch.set_progress_callback(callback)
        orch._emit_progress({"test": True})
        callback.assert_called_once_with({"test": True})

    @pytest.mark.asyncio
    async def test_run_empty_pipeline(self, sample_config, db):
        """Running pipeline with no sources should complete without error."""
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()

        stats = await orch.run(stages=["metadata"])
        assert orch.state == PipelineState.COMPLETED
        assert "metadata" in stats["stages"]

    @pytest.mark.asyncio
    async def test_run_already_running(self, sample_config, db):
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()
        orch.state = PipelineState.RUNNING
        with pytest.raises(RuntimeError, match="already running"):
            await orch.run()

    @pytest.mark.asyncio
    async def test_run_unknown_stage(self, sample_config, db):
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()
        stats = await orch.run(stages=["nonexistent_stage"])
        assert "nonexistent_stage" not in stats["stages"]

    @pytest.mark.asyncio
    async def test_stop_cancels_stage(self, sample_config, db):
        orch = PipelineOrchestrator(config=sample_config, db=db)
        await orch.initialize()
        orch.state = PipelineState.RUNNING
        orch.current_stage = "metadata"
        orch.stop()
        assert orch.state == PipelineState.STOPPING
        assert orch._stages["metadata"]._cancelled is True
