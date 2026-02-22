from __future__ import annotations

import os
import time
from typing import List, Tuple

import cv2

from .exceptions import CameraError


def _preferred_backend_order() -> list[str]:
    raw = os.getenv("FACE_CAMERA_BACKEND_ORDER", "").strip()
    if not raw:
        # Windows laptop webcams are generally more stable on DirectShow.
        if os.name == "nt":
            return ["DirectShow", "Media Foundation", "Auto"]
        return ["Auto", "DirectShow", "Media Foundation"]
    ordered = [item.strip().lower() for item in raw.split(",") if item.strip()]
    mapping = {
        "auto": "Auto",
        "any": "Auto",
        "dshow": "DirectShow",
        "directshow": "DirectShow",
        "msmf": "Media Foundation",
        "mediafoundation": "Media Foundation",
        "media foundation": "Media Foundation",
    }
    result: list[str] = []
    for item in ordered:
        name = mapping.get(item)
        if name and name not in result:
            result.append(name)
    return result or ["Auto", "DirectShow", "Media Foundation"]


def capture_backends() -> List[Tuple[str, int | None]]:
    backend_map: dict[str, int | None] = {
        "Auto": getattr(cv2, "CAP_ANY", None),
        "DirectShow": getattr(cv2, "CAP_DSHOW", None),
        "Media Foundation": getattr(cv2, "CAP_MSMF", None),
    }
    preferred = _preferred_backend_order()
    raw_candidates: List[Tuple[str, int | None]] = []
    for name in preferred:
        raw_candidates.append((name, backend_map.get(name)))
    for fallback_name in ("Auto", "DirectShow", "Media Foundation"):
        if fallback_name not in preferred:
            raw_candidates.append((fallback_name, backend_map.get(fallback_name)))

    candidates: List[Tuple[str, int | None]] = []
    seen: set[int | None] = set()
    for name, backend in raw_candidates:
        if backend in seen:
            continue
        seen.add(backend)
        candidates.append((name, backend))
    return candidates


def open_camera_capture(camera_index: int) -> tuple[cv2.VideoCapture, str]:
    attempted: List[str] = []

    for backend_name, backend in capture_backends():
        attempted.append(backend_name)
        if backend is None:
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index, backend)

        if cap.isOpened():
            # Some backends can report opened=True but fail to deliver frames.
            # Probe a few reads to ensure the stream is actually usable.
            frame_ok = False
            for _ in range(6):
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame_ok = True
                    break
                time.sleep(0.03)

            if frame_ok:
                return cap, backend_name
        cap.release()

    tried = ", ".join(attempted) if attempted else "default backend"
    raise CameraError(
        f"Unable to open webcam index {camera_index}. Tried backends: {tried}."
    )
