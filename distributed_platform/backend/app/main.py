from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api.routes import attendance, auth, devices, employees, events, health, recognitions, sync
from app.core.config import get_settings
from app.core.security import hash_password, safe_decode_token
from app.db.base import Base
from app.db.models import User
from app.db.session import SessionLocal, engine
from app.ws.manager import ws_manager

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("distributed.backend")


def bootstrap_defaults() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.username == settings.bootstrap_admin_username))
        if admin is None:
            admin = User(
                username=settings.bootstrap_admin_username,
                password_hash=hash_password(settings.bootstrap_admin_password),
                role=settings.bootstrap_admin_role,
                is_active=True,
            )
            db.add(admin)
            db.commit()
            logger.info("Created bootstrap admin user '%s'.", settings.bootstrap_admin_username)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_defaults()
    sync.sync_worker.start()
    yield
    sync.sync_worker.stop()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(employees.router, prefix=settings.api_prefix)
app.include_router(devices.router, prefix=settings.api_prefix)
app.include_router(recognitions.router, prefix=settings.api_prefix)
app.include_router(attendance.router, prefix=settings.api_prefix)
app.include_router(events.router, prefix=settings.api_prefix)
app.include_router(sync.router, prefix=settings.api_prefix)


@app.websocket("/ws/events")
async def events_socket(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        return

    payload = safe_decode_token(token)
    if payload is None:
        await websocket.close(code=4401)
        return

    await ws_manager.connect(websocket, channel="global")
    try:
        while True:
            message = await websocket.receive_text()
            if message.lower() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)
        await asyncio.sleep(0)
