from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Employee
from app.services.encryption import embedding_crypto


@dataclass
class MatchResult:
    employee_id: uuid.UUID | None
    employee_name: str | None
    confidence: float
    matched: bool


class FaceMatcher:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._cache: dict[str, tuple[float, np.ndarray, list[tuple[uuid.UUID, str]]]] = {}
        self._cache_ttl_seconds = 20.0

    def _cache_key(self) -> str:
        return "employees-active"

    def _load_embeddings(self, db: Session) -> tuple[np.ndarray, list[tuple[uuid.UUID, str]]]:
        key = self._cache_key()
        cached = self._cache.get(key)
        now = time.time()
        if cached and (now - cached[0]) <= self._cache_ttl_seconds:
            return cached[1], cached[2]

        rows = db.scalars(select(Employee).where(Employee.is_active.is_(True))).all()
        vectors: list[np.ndarray] = []
        metadata: list[tuple[uuid.UUID, str]] = []
        for row in rows:
            vector = embedding_crypto.decrypt(row.embedding_ciphertext)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-9:
                continue
            vectors.append((vector / norm).astype(np.float32))
            metadata.append((row.id, row.name))

        if not vectors:
            matrix = np.empty((0, 512), dtype=np.float32)
        else:
            matrix = np.vstack(vectors).astype(np.float32)
        self._cache[key] = (now, matrix, metadata)
        return matrix, metadata

    def invalidate(self) -> None:
        self._cache.clear()

    def match(self, db: Session, embedding: np.ndarray) -> MatchResult:
        matrix, metadata = self._load_embeddings(db)
        if matrix.size == 0:
            return MatchResult(None, None, 0.0, False)

        scores = matrix @ embedding
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score < self.threshold:
            return MatchResult(None, None, score, False)

        employee_id, employee_name = metadata[idx]
        return MatchResult(employee_id, employee_name, score, True)


@lru_cache(maxsize=1)
def get_matcher() -> FaceMatcher:
    settings = get_settings()
    return FaceMatcher(threshold=settings.match_threshold)
