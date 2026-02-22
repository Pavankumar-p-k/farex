from __future__ import annotations

import threading

import cv2

from arcface_embedder import ArcFaceEmbedder
from config import ClientConfig
from hybrid_api import HybridApiClient
from ws_listener import RecognitionWsListener


def main() -> int:
    cfg = ClientConfig()
    api = HybridApiClient(cfg)
    api.heartbeat()
    print(f"[route] mode={api.route.mode} base={api.route.base_url}")
    print(f"[queue] pending={api.queue.size()} db={cfg.queue_db_path}")

    ws = RecognitionWsListener(target_provider=api.ws_target)
    ws.start()

    thread = threading.Thread(target=api.heartbeat_loop, daemon=True)
    thread.start()

    embedder = ArcFaceEmbedder()
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"[error] camera {cfg.camera_index} failed to open.")
        return 1

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_idx += 1

            embedding = embedder.extract(frame)
            if embedding is not None and (frame_idx % max(1, cfg.match_interval_frames) == 0):
                try:
                    result = api.send_embedding(embedding)
                    if result.get("matched"):
                        print(
                            f"[match] {result.get('employee_name')} "
                            f"confidence={result.get('confidence'):.3f}"
                        )
                except Exception as exc:
                    print(f"[recognition] failed: {exc}")

            cv2.imshow("Windows Edge Client (Q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ws.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
