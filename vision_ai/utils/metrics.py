from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ModulePerf:
    frames: int = 0
    latency_ema_ms: float = 0.0
    latency_samples: int = 0
    events: deque[float] = field(default_factory=lambda: deque(maxlen=240))


class PerformanceTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: dict[str, ModulePerf] = {}

    def update(self, module: str, latency_ms: float) -> None:
        now = time.perf_counter()
        with self._lock:
            stat = self._stats.setdefault(module, ModulePerf())
            stat.frames += 1
            stat.events.append(now)
            stat.latency_samples += 1
            alpha = 0.2
            if stat.latency_samples == 1:
                stat.latency_ema_ms = latency_ms
            else:
                stat.latency_ema_ms = (alpha * latency_ms) + ((1.0 - alpha) * stat.latency_ema_ms)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            output: dict[str, dict[str, float]] = {}
            now = time.perf_counter()
            for module, stat in self._stats.items():
                fps = self._compute_fps(stat.events, now)
                output[module] = {
                    "fps": fps,
                    "latency_ms": stat.latency_ema_ms,
                    "frames": float(stat.frames),
                }
            return output

    @staticmethod
    def _compute_fps(events: deque[float], now: float) -> float:
        if len(events) < 2:
            return 0.0
        window_seconds = max(1e-6, now - events[0])
        return len(events) / window_seconds
