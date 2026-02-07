"""Pipeline orchestrator - coordinates all stages."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from echoharvester.config import Config, get_config
from echoharvester.db import Database, RunStatus, get_db
from echoharvester.stages import (
    DownloadStage,
    ExportStage,
    MetadataStage,
    PreprocessStage,
    ValidateStage,
)

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """Pipeline execution state."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline stages."""

    STAGES = ["metadata", "download", "preprocess", "validate", "export"]

    def __init__(self, config: Config | None = None, db: Database | None = None):
        self.config = config
        self.db = db
        self.state = PipelineState.IDLE
        self.current_stage: str | None = None
        self.run_id: int | None = None

        self._stages: dict[str, Any] = {}
        self._progress_callback: Callable[[dict], None] | None = None
        self._stop_event = asyncio.Event()

    async def initialize(self):
        """Initialize orchestrator with config and database."""
        if self.config is None:
            self.config = get_config()
        if self.db is None:
            self.db = await get_db()

        # Create stage instances
        self._stages = {
            "metadata": MetadataStage(self.config, self.db),
            "download": DownloadStage(self.config, self.db),
            "preprocess": PreprocessStage(self.config, self.db),
            "validate": ValidateStage(self.config, self.db),
            "export": ExportStage(self.config, self.db),
        }

    def set_progress_callback(self, callback: Callable[[dict], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _emit_progress(self, data: dict):
        """Emit progress update."""
        if self._progress_callback:
            self._progress_callback(data)

    async def run(
        self,
        stages: list[str] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Run the pipeline.

        Args:
            stages: List of stages to run (default: all)
            resume: Resume from last checkpoint if available

        Returns:
            Pipeline execution statistics
        """
        if self.state == PipelineState.RUNNING:
            raise RuntimeError("Pipeline is already running")

        await self.initialize()

        stages_to_run = stages or self.STAGES
        self.state = PipelineState.RUNNING
        self._stop_event.clear()

        # Create run record
        self.run_id = await self.db.create_run(
            config_snapshot=self.config.model_dump(mode="json")
        )

        logger.info(f"Starting pipeline run {self.run_id}")
        self._emit_progress({
            "event": "pipeline_start",
            "run_id": self.run_id,
            "stages": stages_to_run,
        })

        stats = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "stages": {},
        }

        try:
            for stage_name in stages_to_run:
                if self._stop_event.is_set():
                    logger.info("Pipeline stop requested")
                    break

                if stage_name not in self._stages:
                    logger.warning(f"Unknown stage: {stage_name}")
                    continue

                self.current_stage = stage_name
                stage = self._stages[stage_name]

                logger.info(f"Running stage: {stage_name}")
                self._emit_progress({
                    "event": "stage_start",
                    "stage": stage_name,
                    "run_id": self.run_id,
                })

                try:
                    stage_stats = await stage.run(self.run_id)
                    stats["stages"][stage_name] = stage_stats

                    self._emit_progress({
                        "event": "stage_complete",
                        "stage": stage_name,
                        "stats": stage_stats,
                    })

                except Exception as e:
                    logger.error(
                        f"Stage {stage_name} failed: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    stats["stages"][stage_name] = {
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                    }
                    self._emit_progress({
                        "event": "stage_error",
                        "stage": stage_name,
                        "error": str(e),
                    })
                    raise

            # Mark run as completed
            self.state = PipelineState.COMPLETED
            await self.db.update_run(
                self.run_id,
                status=RunStatus.COMPLETED.value,
                stats=stats,
                ended=True,
            )

        except Exception as e:
            self.state = PipelineState.ERROR
            await self.db.update_run(
                self.run_id,
                status=RunStatus.ERROR.value,
                stats=stats,
                ended=True,
            )
            raise

        finally:
            self.current_stage = None
            stats["ended_at"] = datetime.now().isoformat()

            self._emit_progress({
                "event": "pipeline_complete",
                "run_id": self.run_id,
                "state": self.state.value,
                "stats": stats,
            })

        return stats

    async def run_stage(self, stage_name: str) -> dict[str, Any]:
        """Run a single stage.

        Args:
            stage_name: Name of the stage to run

        Returns:
            Stage execution statistics
        """
        return await self.run(stages=[stage_name])

    def stop(self):
        """Request pipeline stop."""
        if self.state != PipelineState.RUNNING:
            return

        logger.info("Stop requested")
        self.state = PipelineState.STOPPING
        self._stop_event.set()

        # Cancel current stage
        if self.current_stage and self.current_stage in self._stages:
            self._stages[self.current_stage].cancel()

    async def get_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        progress = []
        if self.run_id:
            progress = await self.db.get_progress(self.run_id)

        latest_run = await self.db.get_latest_run()

        return {
            "state": self.state.value,
            "current_stage": self.current_stage,
            "run_id": self.run_id,
            "progress": progress,
            "latest_run": latest_run,
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get overall pipeline statistics."""
        return await self.db.get_stats()


# Global orchestrator instance
_orchestrator: PipelineOrchestrator | None = None


async def get_orchestrator() -> PipelineOrchestrator:
    """Get or create global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator
