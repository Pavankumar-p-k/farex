from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class FaceIdentity:
    employee_id: str
    name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    new_attendance: bool = False


@dataclass
class FaceResult:
    frame_id: int
    identities: list[FaceIdentity] = field(default_factory=list)
    registration_mode: bool = False
    registration_progress: float = 0.0
    message: str = ""


@dataclass
class ObjectItem:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class ObjectResult:
    frame_id: int
    detections: list[ObjectItem] = field(default_factory=list)


@dataclass
class OCRItem:
    text: str
    confidence: float
    polygon: list[tuple[int, int]]


@dataclass
class OCRResult:
    frame_id: int
    entries: list[OCRItem] = field(default_factory=list)


@dataclass
class WorkerEnvelope:
    module: str
    frame_id: int
    payload: Any
