from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, role: str, device_id: str | None = None) -> str:
    settings = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    payload = {
        "sub": subject,
        "role": role,
        "device_id": device_id,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


def require_role(actual_role: str, allowed_roles: Iterable[str]) -> None:
    if actual_role not in set(allowed_roles):
        raise PermissionError(f"Role '{actual_role}' is not allowed for this action.")


def normalize_embedding(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("Embedding must be a 1D vector.")
    if vector.size != 512:
        raise ValueError("ArcFace embedding must contain exactly 512 values.")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Embedding norm is zero.")
    return vector / norm


def safe_decode_token(token: str) -> dict | None:
    try:
        return decode_access_token(token)
    except JWTError:
        return None

