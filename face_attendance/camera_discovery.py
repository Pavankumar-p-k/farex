from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Set

import cv2

from .camera_capture import capture_backends


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class CameraProbeResult:
    index: int
    width: int
    height: int
    fps: float
    backend: str = "unknown"


def discover_cameras(
    max_index: int = 8,
    exclude_indices: Set[int] | None = None,
) -> List[CameraProbeResult]:
    excluded = exclude_indices or set()
    cameras: List[CameraProbeResult] = []
    for camera_index in range(max_index + 1):
        if camera_index in excluded:
            continue
        detected: CameraProbeResult | None = None
        for candidate_name, backend in capture_backends():
            if backend is None:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            ok = False
            frame = None
            for _ in range(6):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.02)

            if ok and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                detected = CameraProbeResult(
                    index=camera_index,
                    width=width,
                    height=height,
                    fps=fps,
                    backend=candidate_name,
                )
                cap.release()
                break

            cap.release()

        if detected is not None:
            cameras.append(detected)

    return cameras


def pick_camera_index(
    preferred_index: int,
    available: List[CameraProbeResult],
    prefer_highest_index: bool = True,
) -> int:
    if not available:
        return preferred_index

    available_indices = [entry.index for entry in available]
    if preferred_index in available_indices:
        return preferred_index

    if prefer_highest_index:
        return max(available_indices)
    return min(available_indices)


def camera_autoselect_enabled() -> bool:
    return _bool_env("FACE_CAMERA_AUTOSELECT", True)
