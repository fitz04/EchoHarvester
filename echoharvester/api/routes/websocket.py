"""WebSocket routes for real-time updates."""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]):
        """Broadcast message to all connections."""
        if not self.active_connections:
            return

        data = json.dumps(message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal(self, websocket: WebSocket, message: dict[str, Any]):
        """Send message to a specific connection."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception:
            self.disconnect(websocket)


# Global connection manager
manager = ConnectionManager()


def get_progress_callback():
    """Get a callback function for pipeline progress updates."""

    async def callback(data: dict):
        await manager.broadcast(data)

    return callback


@router.websocket("/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for pipeline progress updates."""
    await manager.connect(websocket)

    # Set up progress callback on orchestrator
    orchestrator = websocket.app.state.orchestrator
    orchestrator.set_progress_callback(
        lambda data: asyncio.create_task(manager.broadcast(data))
    )

    try:
        # Send initial status
        status = await orchestrator.get_status()
        await manager.send_personal(websocket, {
            "event": "connected",
            "status": status,
        })

        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # Ping every 30 seconds
                )

                # Handle client messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await manager.send_personal(websocket, {"type": "pong"})
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await manager.send_personal(websocket, {"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for log streaming."""
    await manager.connect(websocket)

    # Create a custom log handler that broadcasts
    class WebSocketHandler(logging.Handler):
        def emit(self, record):
            try:
                log_entry = {
                    "event": "log",
                    "level": record.levelname,
                    "message": self.format(record),
                    "logger": record.name,
                    "timestamp": record.created,
                }
                asyncio.create_task(manager.send_personal(websocket, log_entry))
            except Exception:
                pass

    handler = WebSocketHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Add handler to root logger
    root_logger = logging.getLogger("echoharvester")
    root_logger.addHandler(handler)

    try:
        await manager.send_personal(websocket, {
            "event": "connected",
            "message": "Log streaming started",
        })

        while True:
            try:
                await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                await manager.send_personal(websocket, {"type": "ping"})

    except WebSocketDisconnect:
        pass
    finally:
        root_logger.removeHandler(handler)
        manager.disconnect(websocket)
