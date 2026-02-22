from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[int, int]
Polygon = List[Point]


@dataclass
class OCRDetection:
    polygon: Polygon
    text: str
    confidence: float
    angle_degrees: float
