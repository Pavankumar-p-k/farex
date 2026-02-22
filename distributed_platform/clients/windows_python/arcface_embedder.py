from __future__ import annotations

import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None


class ArcFaceEmbedder:
    def __init__(self) -> None:
        if FaceAnalysis is None:
            raise RuntimeError("insightface is required for ArcFace embeddings.")
        self.app = FaceAnalysis(name="buffalo_l")
        # CUDAExecutionProvider will be used if onnxruntime-gpu is installed and GPU is available.
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract(self, frame_bgr) -> list[float] | None:
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        # Highest detection score face first.
        face = sorted(faces, key=lambda item: float(item.det_score), reverse=True)[0]
        emb = np.asarray(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-9:
            return None
        emb = emb / norm
        return emb.astype(np.float32).tolist()

