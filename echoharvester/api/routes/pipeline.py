"""Pipeline control API routes."""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class PipelineStartRequest(BaseModel):
    """Request model for starting pipeline."""

    stages: list[str] | None = None
    resume: bool = True


@router.get("/status")
async def get_status(request: Request) -> dict[str, Any]:
    """Get current pipeline status."""
    orchestrator = request.app.state.orchestrator
    return await orchestrator.get_status()


@router.post("/start")
async def start_pipeline(
    body: PipelineStartRequest,
    request: Request,
) -> dict[str, str]:
    """Start the pipeline."""
    orchestrator = request.app.state.orchestrator

    if orchestrator.state.value == "running":
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    # Run in background
    async def run_pipeline():
        try:
            await orchestrator.run(stages=body.stages, resume=body.resume)
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")

    asyncio.create_task(run_pipeline())

    return {"message": "Pipeline started"}


@router.post("/stop")
async def stop_pipeline(request: Request) -> dict[str, str]:
    """Stop the running pipeline."""
    orchestrator = request.app.state.orchestrator

    if orchestrator.state.value != "running":
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    orchestrator.stop()
    return {"message": "Stop requested"}


@router.post("/stage/{stage_name}")
async def run_stage(
    stage_name: str,
    request: Request,
) -> dict[str, str]:
    """Run a specific stage."""
    orchestrator = request.app.state.orchestrator

    valid_stages = ["metadata", "download", "preprocess", "validate", "export"]
    if stage_name not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {valid_stages}",
        )

    if orchestrator.state.value == "running":
        raise HTTPException(status_code=400, detail="Pipeline is already running")

    async def run_single_stage():
        try:
            await orchestrator.run_stage(stage_name)
        except Exception as e:
            logger.exception(f"Stage error: {e}")

    asyncio.create_task(run_single_stage())

    return {"message": f"Stage {stage_name} started"}


@router.get("/runs")
async def list_runs(request: Request, limit: int = 10) -> list[dict[str, Any]]:
    """List recent pipeline runs."""
    db = request.app.state.db
    cursor = await db.conn.execute(
        "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


@router.get("/runs/{run_id}")
async def get_run(run_id: int, request: Request) -> dict[str, Any]:
    """Get a specific pipeline run."""
    db = request.app.state.db
    run = await db.get_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    progress = await db.get_progress(run_id)
    run["progress"] = progress

    return run
