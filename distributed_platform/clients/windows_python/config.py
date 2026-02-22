from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClientConfig:
    device_name: str = os.getenv("CLIENT_DEVICE_NAME", "win-frontdesk-01")
    device_type: str = "windows"
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:9000")
    cloud_base_url: str = os.getenv("CLOUD_BASE_URL", "https://YOUR_PROJECT.supabase.co/functions/v1")
    login_username: str = os.getenv("CLIENT_USERNAME", "admin")
    login_password: str = os.getenv("CLIENT_PASSWORD", "ChangeMe123!")
    ws_local_url: str = os.getenv("WS_LOCAL_URL", "ws://127.0.0.1:9000/ws/events")
    ws_cloud_url: str = os.getenv("WS_CLOUD_URL", "wss://YOUR_PROJECT.supabase.co/realtime/v1/websocket")
    camera_index: int = int(os.getenv("CLIENT_CAMERA_INDEX", "0"))
    heartbeat_seconds: int = int(os.getenv("CLIENT_HEARTBEAT_SECONDS", "20"))
    match_interval_frames: int = int(os.getenv("CLIENT_MATCH_INTERVAL_FRAMES", "3"))
    request_timeout_seconds: float = float(os.getenv("CLIENT_REQUEST_TIMEOUT_SECONDS", "8.0"))
    health_timeout_seconds: float = float(os.getenv("CLIENT_HEALTH_TIMEOUT_SECONDS", "1.2"))
    queue_max_items: int = int(os.getenv("CLIENT_QUEUE_MAX_ITEMS", "5000"))
    queue_flush_batch: int = int(os.getenv("CLIENT_QUEUE_FLUSH_BATCH", "20"))
    queue_db_path: str = os.getenv(
        "CLIENT_QUEUE_DB_PATH",
        str(Path(__file__).resolve().parent / "state" / "offline_embeddings.db"),
    )
