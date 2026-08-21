"""WebSocket server for real-time training metric streaming.

Architecture:
    Training loop → MetricsBroadcaster.publish() → WebSocket clients

The broadcaster uses an in-memory pub/sub pattern. Multiple dashboard
tabs can connect and receive the same live metrics stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MetricsBroadcaster:
    """Pub/sub broadcaster for training metrics.

    Training callbacks publish metrics here, and all connected
    WebSocket clients receive them in real-time.
    """

    _instance: MetricsBroadcaster | None = None

    def __init__(self) -> None:
        # experiment_id → set of connected WebSocket clients
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> MetricsBroadcaster:
        """Get the global broadcaster singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    async def connect(self, experiment_id: str, websocket: WebSocket) -> None:
        """Register a new WebSocket client for an experiment."""
        await websocket.accept()
        async with self._lock:
            if experiment_id not in self._connections:
                self._connections[experiment_id] = set()
            self._connections[experiment_id].add(websocket)
        logger.info("WebSocket connected for experiment %s", experiment_id)

    async def disconnect(self, experiment_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket client."""
        async with self._lock:
            if experiment_id in self._connections:
                self._connections[experiment_id].discard(websocket)
                if not self._connections[experiment_id]:
                    del self._connections[experiment_id]
        logger.info("WebSocket disconnected for experiment %s", experiment_id)

    async def broadcast(self, experiment_id: str, data: dict[str, Any]) -> None:
        """Send metrics to all clients watching an experiment."""
        async with self._lock:
            clients = self._connections.get(experiment_id, set()).copy()

        if not clients:
            return

        message = json.dumps(data)
        dead_clients: list[WebSocket] = []

        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead_clients.append(ws)

        # Clean up dead connections
        if dead_clients:
            async with self._lock:
                for ws in dead_clients:
                    if experiment_id in self._connections:
                        self._connections[experiment_id].discard(ws)

    def publish_sync(self, experiment_id: str, data: dict[str, Any]) -> None:
        """Publish from synchronous code (training callbacks).

        Creates a new event loop task if one is running, otherwise
        uses asyncio.run() for one-shot publishing.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(experiment_id, data))
        except RuntimeError:
            # No event loop running — fire-and-forget
            try:
                asyncio.run(self.broadcast(experiment_id, data))
            except Exception:
                pass  # Best-effort, don't crash the training loop

    @property
    def active_experiments(self) -> list[str]:
        """List experiment IDs with active WebSocket connections."""
        return list(self._connections.keys())


def register_websocket(app: FastAPI) -> None:
    """Register the WebSocket endpoint on the FastAPI app."""

    @app.websocket("/ws/training/{experiment_id}")
    async def training_ws(websocket: WebSocket, experiment_id: str) -> None:
        """WebSocket endpoint for live training metrics.

        Clients connect here to receive real-time metrics updates
        for a specific training run.
        """
        broadcaster = MetricsBroadcaster.get_instance()
        await broadcaster.connect(experiment_id, websocket)

        try:
            while True:
                # Keep connection alive — wait for client messages
                # (ping/pong, or client can send control messages)
                data = await websocket.receive_text()
                # Handle client messages if needed (e.g., "pause", "resume")
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect:
            await broadcaster.disconnect(experiment_id, websocket)
        except Exception:
            await broadcaster.disconnect(experiment_id, websocket)
