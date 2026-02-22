from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str = "global") -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[channel].add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            for channel in list(self._connections.keys()):
                self._connections[channel].discard(websocket)
                if not self._connections[channel]:
                    del self._connections[channel]

    async def broadcast(self, message: dict[str, Any], channel: str = "global") -> None:
        async with self._lock:
            targets = list(self._connections.get(channel, set()))
        stale: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            await self.disconnect(ws)


ws_manager = ConnectionManager()

