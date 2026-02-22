from __future__ import annotations

import base64

import numpy as np
from cryptography.fernet import Fernet

from app.core.config import get_settings


class EmbeddingCrypto:
    def __init__(self, key_material: str):
        padded = key_material.encode("utf-8")
        key = base64.urlsafe_b64encode(padded.ljust(32, b"0")[:32])
        self._fernet = Fernet(key)

    def encrypt(self, vector: np.ndarray) -> str:
        payload = vector.astype(np.float32).tobytes()
        return self._fernet.encrypt(payload).decode("utf-8")

    def decrypt(self, ciphertext: str) -> np.ndarray:
        payload = self._fernet.decrypt(ciphertext.encode("utf-8"))
        return np.frombuffer(payload, dtype=np.float32)


embedding_crypto = EmbeddingCrypto(get_settings().embedding_cipher_key)

