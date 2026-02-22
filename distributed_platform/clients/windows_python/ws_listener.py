from __future__ import annotations

import json
import threading
import time
from typing import Callable

import websocket


class RecognitionWsListener:
    def __init__(self, target_provider: Callable[[], tuple[str, str]], reconnect_seconds: int = 3) -> None:
        self.target_provider = target_provider
        self.reconnect_seconds = max(1, reconnect_seconds)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._app: websocket.WebSocketApp | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-listener")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._app is not None:
            try:
                self._app.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            ws_url, token = self.target_provider()
            if not ws_url or not token:
                time.sleep(self.reconnect_seconds)
                continue

            target = f"{ws_url}?token={token}"
            print(f"[ws] connecting {ws_url}")

            def _on_message(_ws, message: str) -> None:
                try:
                    payload = json.loads(message)
                except Exception:
                    payload = {"raw": message}
                print(f"[ws] {payload}")

            def _on_error(_ws, error) -> None:
                print(f"[ws] error: {error}")

            def _on_close(_ws, status_code, msg) -> None:
                print(f"[ws] closed code={status_code} msg={msg}")

            self._app = websocket.WebSocketApp(
                target,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            self._app.run_forever(ping_interval=20, ping_timeout=5)
            self._app = None

            if not self._stop_event.is_set():
                time.sleep(self.reconnect_seconds)

