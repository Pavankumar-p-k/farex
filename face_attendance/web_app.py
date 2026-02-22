import time
import logging
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse
import socket

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .config import BASE_DIR, CAMERA_INDEX, DB_PATH, DEVICE, RECOGNITION_THRESHOLD
from .database import AttendanceDatabase
from .exceptions import AttendanceError
from .face_engine import FaceEngine
from .web_runtime import WebVisionRuntime


WEB_DIR = Path(BASE_DIR) / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
logger = logging.getLogger("face_attendance.web_app")


class RegisterPendingBody(BaseModel):
    employee_id: str
    name: str


class SelectCameraBody(BaseModel):
    camera_index: int
    pin_manual: bool = True


class CameraAutoSelectBody(BaseModel):
    enabled: bool


def _mjpeg_frame_generator(runtime: WebVisionRuntime, stream_name: str) -> Iterator[bytes]:
    while True:
        frame = runtime.get_jpeg_frame(stream_name=stream_name)
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


def _resolve_mobile_urls(base_url: str) -> list[str]:
    parsed = urlparse(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    addresses = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                addresses.add(ip)
    except Exception:
        pass

    if not addresses:
        addresses.add("127.0.0.1")

    scheme = parsed.scheme or "http"
    return [f"{scheme}://{ip}:{port}" for ip in sorted(addresses)]


def create_web_app(camera_index: int | None = None) -> FastAPI:
    app = FastAPI(title="Hologram Vision Dashboard", version="1.0.0")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    db = AttendanceDatabase(DB_PATH)
    engine = FaceEngine(device=DEVICE)
    runtime = WebVisionRuntime(
        db=db,
        engine=engine,
        threshold=RECOGNITION_THRESHOLD,
        camera_index=CAMERA_INDEX if camera_index is None else int(camera_index),
    )

    @app.on_event("startup")
    def _startup() -> None:
        try:
            runtime.start()
        except Exception as exc:
            runtime.last_error = f"Runtime startup failed: {exc}"
            logger.exception("Web runtime startup failed")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        runtime.stop()

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/api/stream/camera")
    def camera_stream():
        return StreamingResponse(
            _mjpeg_frame_generator(runtime, stream_name="camera"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/stream/skeleton")
    def skeleton_stream():
        return StreamingResponse(
            _mjpeg_frame_generator(runtime, stream_name="skeleton"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/state")
    def state():
        return JSONResponse(runtime.get_dashboard_state())

    @app.post("/api/register-pending")
    def register_pending(payload: RegisterPendingBody):
        try:
            saved = runtime.register_pending_unknown(
                employee_id=payload.employee_id,
                name=payload.name,
            )
            return {"ok": True, "saved": saved}
        except AttendanceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/clear-pending")
    def clear_pending():
        runtime.clear_pending_unknown()
        return {"ok": True}

    @app.get("/api/cameras")
    def list_cameras(refresh: bool = False):
        return runtime.list_cameras(refresh=refresh)

    @app.post("/api/camera/select")
    def select_camera(payload: SelectCameraBody):
        try:
            return runtime.switch_camera(payload.camera_index, pin_manual=payload.pin_manual)
        except AttendanceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/camera/auto-select")
    def set_camera_auto_select(payload: CameraAutoSelectBody):
        return runtime.set_camera_auto_select(payload.enabled)

    @app.get("/api/mobile-access")
    def mobile_access(request: Request):
        return {
            "urls": _resolve_mobile_urls(str(request.base_url)),
            "hint": "Open one URL on your phone while laptop and phone are on same Wi-Fi/USB tethering.",
        }

    return app


app = create_web_app()
