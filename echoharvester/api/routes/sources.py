"""Source management API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class SourceCreate(BaseModel):
    """Request model for creating a source."""

    type: str
    url_or_path: str
    label: str = ""
    config: dict | None = None


class SourceResponse(BaseModel):
    """Response model for a source."""

    id: int
    type: str
    url_or_path: str
    label: str
    status: str
    total_items: int
    processed_items: int
    created_at: str


@router.get("")
async def list_sources(request: Request) -> list[dict[str, Any]]:
    """List all sources."""
    db = request.app.state.db
    sources = await db.get_sources()
    return sources


@router.post("")
async def create_source(source: SourceCreate, request: Request) -> dict[str, Any]:
    """Create a new source."""
    db = request.app.state.db

    source_id = await db.add_source(
        source_type=source.type,
        url_or_path=source.url_or_path,
        label=source.label,
        config=source.config,
    )

    return {"id": source_id, "message": "Source created"}


@router.get("/{source_id}")
async def get_source(source_id: int, request: Request) -> dict[str, Any]:
    """Get a source by ID."""
    db = request.app.state.db
    source = await db.get_source(source_id)

    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    return source


@router.delete("/{source_id}")
async def delete_source(source_id: int, request: Request) -> dict[str, str]:
    """Delete a source."""
    db = request.app.state.db

    source = await db.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    await db.delete_source(source_id)
    return {"message": "Source deleted"}


@router.get("/{source_id}/items")
async def list_source_items(
    source_id: int,
    request: Request,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List media items for a source."""
    db = request.app.state.db
    items = await db.get_media_items(source_id=source_id, status=status, limit=limit)
    return items
