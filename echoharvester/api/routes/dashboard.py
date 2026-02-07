"""Dashboard API routes."""

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

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


SORT_WHITELIST = {"cer", "duration_sec", "snr_db", "segment_index", "created_at"}


@router.get("/segments")
async def list_segments(
    request: Request,
    status: str | None = None,
    media_id: str | None = None,
    cer_min: float | None = None,
    cer_max: float | None = None,
    duration_min: float | None = None,
    duration_max: float | None = None,
    snr_min: float | None = None,
    snr_max: float | None = None,
    text_search: str | None = None,
    reject_reason: str | None = None,
    sort_by: str | None = None,
    sort_order: str = "asc",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List segments with filtering."""
    db = request.app.state.db

    where = "WHERE 1=1"
    params: list[Any] = []

    if status:
        where += " AND status = ?"
        params.append(status)
    if media_id:
        where += " AND media_id = ?"
        params.append(media_id)
    if cer_min is not None:
        where += " AND cer >= ?"
        params.append(cer_min)
    if cer_max is not None:
        where += " AND cer <= ?"
        params.append(cer_max)
    if duration_min is not None:
        where += " AND duration_sec >= ?"
        params.append(duration_min)
    if duration_max is not None:
        where += " AND duration_sec <= ?"
        params.append(duration_max)
    if snr_min is not None:
        where += " AND snr_db >= ?"
        params.append(snr_min)
    if snr_max is not None:
        where += " AND snr_db <= ?"
        params.append(snr_max)
    if text_search:
        where += " AND (normalized_text LIKE ? OR original_text LIKE ?)"
        pattern = f"%{text_search}%"
        params.extend([pattern, pattern])
    if reject_reason:
        where += " AND reject_reason LIKE ?"
        params.append(f"{reject_reason}%")

    # Count query
    cursor = await db.conn.execute(
        f"SELECT COUNT(*) as count FROM segments {where}", params
    )
    total = (await cursor.fetchone())["count"]

    # Data query with sorting
    order = "ORDER BY media_id, segment_index"
    if sort_by and sort_by in SORT_WHITELIST:
        direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        order = f"ORDER BY {sort_by} {direction}"

    cursor = await db.conn.execute(
        f"SELECT * FROM segments {where} {order} LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    segments = [dict(row) for row in await cursor.fetchall()]

    return {
        "segments": segments,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/segments/filter-options")
async def get_filter_options(request: Request) -> dict[str, Any]:
    """Get available filter options for the explore page."""
    db = request.app.state.db

    # Distinct media IDs
    cursor = await db.conn.execute(
        "SELECT DISTINCT media_id FROM segments ORDER BY media_id"
    )
    media_ids = [row["media_id"] for row in await cursor.fetchall()]

    # Distinct reject reasons (normalize to prefix before ':')
    cursor = await db.conn.execute(
        "SELECT DISTINCT reject_reason FROM segments WHERE reject_reason IS NOT NULL AND reject_reason != ''"
    )
    raw_reasons = [row["reject_reason"] for row in await cursor.fetchall()]
    reason_prefixes = sorted({r.split(":")[0] for r in raw_reasons})

    # Value ranges
    cursor = await db.conn.execute(
        """
        SELECT
            MIN(cer) as cer_min, MAX(cer) as cer_max,
            MIN(duration_sec) as duration_min, MAX(duration_sec) as duration_max,
            MIN(snr_db) as snr_min, MAX(snr_db) as snr_max
        FROM segments
        WHERE cer IS NOT NULL OR duration_sec IS NOT NULL OR snr_db IS NOT NULL
        """
    )
    ranges = dict(await cursor.fetchone())

    return {
        "media_ids": media_ids,
        "reject_reasons": reason_prefixes,
        "ranges": ranges,
    }


@router.get("/segments/{segment_id}/audio")
async def get_segment_audio(segment_id: str, request: Request) -> FileResponse:
    """Serve segment audio file."""
    db = request.app.state.db

    cursor = await db.conn.execute(
        "SELECT audio_path FROM segments WHERE id = ?",
        (segment_id,),
    )
    row = await cursor.fetchone()

    if not row or not row["audio_path"]:
        raise HTTPException(status_code=404, detail="Audio not found")

    audio_path = Path.cwd() / row["audio_path"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")


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


# ==================== Transcribe Endpoints ====================


@router.get("/transcribe/media")
async def transcribe_media_list(request: Request) -> list[dict[str, Any]]:
    """Get media list with review progress for transcribe page."""
    db = request.app.state.db
    cursor = await db.conn.execute(
        """
        SELECT
            m.id, m.title, m.duration_sec,
            COUNT(s.id) as total,
            SUM(CASE WHEN s.status = 'approved' THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN s.status IN ('gpu_pass', 'exported') THEN 1 ELSE 0 END) as pass_count,
            SUM(CASE WHEN s.status LIKE '%reject%' THEN 1 ELSE 0 END) as reject_count
        FROM media_items m
        JOIN segments s ON s.media_id = m.id
        GROUP BY m.id
        HAVING total > 0
        ORDER BY m.title
        """
    )
    return [dict(row) for row in await cursor.fetchall()]


@router.get("/transcribe/segments/{media_id}")
async def transcribe_segments(
    media_id: str,
    request: Request,
    status_filter: str = "all",
) -> list[dict[str, Any]]:
    """Get segments for a media item for transcribe page."""
    db = request.app.state.db

    where = "WHERE s.media_id = ?"
    params: list[Any] = [media_id]

    if status_filter == "needs_review":
        where += " AND s.status IN ('gpu_pass', 'gpu_reject', 'exported')"
    elif status_filter == "approved":
        where += " AND s.status = 'approved'"

    cursor = await db.conn.execute(
        f"""
        SELECT s.*, m.title as media_title
        FROM segments s
        JOIN media_items m ON s.media_id = m.id
        {where}
        ORDER BY s.segment_index
        """,
        params,
    )
    return [dict(row) for row in await cursor.fetchall()]


class ApproveRequest(BaseModel):
    normalized_text: str
    source: str = "manual_edit"  # "manual_edit" | "use_asr"


@router.put("/transcribe/segments/{segment_id}/approve")
async def approve_segment(
    segment_id: str,
    body: ApproveRequest,
    request: Request,
) -> dict[str, Any]:
    """Approve a segment with corrected text."""
    from echoharvester.utils.cer import calculate_cer

    db = request.app.state.db

    cursor = await db.conn.execute(
        "SELECT * FROM segments WHERE id = ?", (segment_id,)
    )
    seg = await cursor.fetchone()
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Recalculate CER: reference=normalized_text, hypothesis=whisper_text
    new_cer = None
    if seg["whisper_text"]:
        new_cer = calculate_cer(body.normalized_text, seg["whisper_text"])

    await db.conn.execute(
        """
        UPDATE segments
        SET normalized_text = ?,
            cer = ?,
            status = 'approved',
            approved_at = ?,
            approved_by = ?
        WHERE id = ?
        """,
        (body.normalized_text, new_cer, str(datetime.now()), body.source, segment_id),
    )
    await db.conn.commit()

    return {
        "id": segment_id,
        "status": "approved",
        "normalized_text": body.normalized_text,
        "cer": new_cer,
        "source": body.source,
    }


class RejectRequest(BaseModel):
    reason: str = "unintelligible"


@router.put("/transcribe/segments/{segment_id}/reject")
async def reject_segment(
    segment_id: str,
    body: RejectRequest,
    request: Request,
) -> dict[str, Any]:
    """Reject a segment during manual review."""
    db = request.app.state.db

    cursor = await db.conn.execute(
        "SELECT id FROM segments WHERE id = ?", (segment_id,)
    )
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="Segment not found")

    await db.conn.execute(
        """
        UPDATE segments
        SET status = 'gpu_reject',
            reject_reason = ?
        WHERE id = ?
        """,
        (f"manual_review:{body.reason}", segment_id),
    )
    await db.conn.commit()

    return {"id": segment_id, "status": "gpu_reject", "reject_reason": f"manual_review:{body.reason}"}
