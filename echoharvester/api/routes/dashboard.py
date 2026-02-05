"""Dashboard API routes."""

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    """Get overall pipeline statistics."""
    db = request.app.state.db
    return await db.get_stats()


@router.get("/recent")
async def get_recent(request: Request, limit: int = 20) -> list[dict[str, Any]]:
    """Get recently processed segments."""
    db = request.app.state.db
    return await db.get_recent_segments(limit=limit)


@router.get("/segments")
async def list_segments(
    request: Request,
    status: str | None = None,
    media_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List segments with filtering."""
    db = request.app.state.db

    segments = await db.get_segments(
        media_id=media_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    # Get total count for pagination
    count_query = "SELECT COUNT(*) as count FROM segments WHERE 1=1"
    params = []
    if status:
        count_query += " AND status = ?"
        params.append(status)
    if media_id:
        count_query += " AND media_id = ?"
        params.append(media_id)

    cursor = await db.conn.execute(count_query, params)
    total = (await cursor.fetchone())["count"]

    return {
        "segments": segments,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/segments/{segment_id}")
async def get_segment(segment_id: str, request: Request) -> dict[str, Any]:
    """Get a specific segment."""
    db = request.app.state.db

    cursor = await db.conn.execute(
        """
        SELECT s.*, m.title as media_title, m.source_type
        FROM segments s
        JOIN media_items m ON s.media_id = m.id
        WHERE s.id = ?
        """,
        (segment_id,),
    )
    row = await cursor.fetchone()

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Segment not found")

    return dict(row)


@router.get("/media")
async def list_media(
    request: Request,
    status: str | None = None,
    source_id: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List media items."""
    db = request.app.state.db
    return await db.get_media_items(source_id=source_id, status=status, limit=limit)


@router.get("/media/{media_id}")
async def get_media(media_id: str, request: Request) -> dict[str, Any]:
    """Get a specific media item."""
    db = request.app.state.db
    item = await db.get_media_item(media_id)

    if not item:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Media item not found")

    # Get segment counts
    cursor = await db.conn.execute(
        """
        SELECT status, COUNT(*) as count
        FROM segments
        WHERE media_id = ?
        GROUP BY status
        """,
        (media_id,),
    )
    segment_counts = {row["status"]: row["count"] for row in await cursor.fetchall()}
    item["segment_counts"] = segment_counts

    return item
