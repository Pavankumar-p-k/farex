from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

BBox = Tuple[int, int, int, int]


@dataclass
class ObjectDetection:
    class_id: int
    label: str
    confidence: float
    bbox: BBox
