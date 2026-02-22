from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from config import ClientConfig
from offline_queue import OfflineEmbeddingQueue


@dataclass(frozen=True)
class RoutingDecision:
    mode: str
    base_url: str
    ws_url: str


class HybridApiClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self._lock = threading.RLock()
        self._session_lock = threading.Lock()
        self._tokens: dict[str, str] = {}
        self._device_ids: dict[str, str] = {}
        self.queue = OfflineEmbeddingQueue(db_path=cfg.queue_db_path, max_items=cfg.queue_max_items)
        self.route = self._choose_route()

    @property
    def token(self) -> str | None:
        with self._lock:
            return self._tokens.get(self.route.base_url)

    @property
    def device_id(self) -> str | None:
        with self._lock:
            return self._device_ids.get(self.route.base_url)

    def _local_route(self) -> RoutingDecision:
        return RoutingDecision(mode="local", base_url=self.cfg.local_base_url, ws_url=self.cfg.ws_local_url)

    def _cloud_route(self) -> RoutingDecision:
        return RoutingDecision(mode="cloud", base_url=self.cfg.cloud_base_url, ws_url=self.cfg.ws_cloud_url)

    def _is_local_healthy(self) -> bool:
        health_url = f"{self.cfg.local_base_url}/api/v1/health"
        try:
            resp = requests.get(health_url, timeout=self.cfg.health_timeout_seconds)
            return bool(resp.ok)
        except Exception:
            return False

    def _choose_route(self) -> RoutingDecision:
        if self._is_local_healthy():
            return self._local_route()
        return self._cloud_route()

    def _other_route(self, route: RoutingDecision) -> RoutingDecision:
        if route.mode == "local":
            return self._cloud_route()
        return self._local_route()

    def _set_route(self, route: RoutingDecision, reason: str) -> None:
        with self._lock:
            previous = self.route
            self.route = route
        if previous.base_url != route.base_url:
            print(f"[route] switch {previous.mode} -> {route.mode} ({reason})")

    def _login_on_route(self, route: RoutingDecision, force: bool = False) -> str:
        with self._lock:
            if not force and route.base_url in self._tokens:
                return self._tokens[route.base_url]

        url = f"{route.base_url}/api/v1/auth/token"
        with self._session_lock:
            resp = self.session.post(
                url,
                json={"username": self.cfg.login_username, "password": self.cfg.login_password},
                timeout=self.cfg.request_timeout_seconds,
            )
        resp.raise_for_status()
        token = str(resp.json()["access_token"])
        with self._lock:
            self._tokens[route.base_url] = token
        return token

    def _post_json(
        self,
        route: RoutingDecision,
        path: str,
        payload: dict[str, Any],
        retry_auth: bool = True,
    ) -> dict[str, Any]:
        token = self._login_on_route(route)
        url = f"{route.base_url}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        with self._session_lock:
            resp = self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.cfg.request_timeout_seconds,
            )
        if resp.status_code == 401 and retry_auth:
            self._login_on_route(route, force=True)
            return self._post_json(route=route, path=path, payload=payload, retry_auth=False)
        resp.raise_for_status()
        return resp.json()

    def login(self) -> None:
        route = self._choose_route()
        self._set_route(route, reason="login-probe")
        self._login_on_route(self.route, force=True)

    def _heartbeat_on_route(self, route: RoutingDecision) -> str:
        with self._lock:
            current_device = self._device_ids.get(route.base_url)
        payload = {
            "device_id": current_device,
            "device_name": self.cfg.device_name,
            "device_type": self.cfg.device_type,
            "network_mode": route.mode,
            "status": "online",
            "metadata": {"client": "windows_python", "version": "1.1.0"},
        }
        body = self._post_json(route=route, path="/api/v1/devices/heartbeat", payload=payload)
        device_id = str(body["id"])
        with self._lock:
            self._device_ids[route.base_url] = device_id
        return device_id

    def heartbeat(self) -> None:
        route = self._choose_route()
        self._set_route(route, reason="heartbeat-probe")
        self._heartbeat_on_route(self.route)

    def _build_recognition_payload(self, embedding: list[float]) -> dict[str, Any]:
        return {
            "embedding": embedding,
            "metadata": {
                "client": "windows_python",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _send_embedding_on_route(self, route: RoutingDecision, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            known_device_id = self._device_ids.get(route.base_url)
        if not known_device_id:
            known_device_id = self._heartbeat_on_route(route)

        body = dict(payload)
        body["device_id"] = known_device_id
        try:
            return self._post_json(route=route, path="/api/v1/recognitions/match", payload=body)
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 404:
                known_device_id = self._heartbeat_on_route(route)
                body["device_id"] = known_device_id
                return self._post_json(route=route, path="/api/v1/recognitions/match", payload=body)
            raise

    def _send_with_failover(self, payload: dict[str, Any]) -> dict[str, Any]:
        preferred = self._choose_route()
        self._set_route(preferred, reason="recognition-probe")
        routes = [preferred, self._other_route(preferred)]
        errors: list[str] = []

        for route in routes:
            if route.mode == "local" and not self._is_local_healthy():
                errors.append("local route unhealthy")
                continue
            try:
                result = self._send_embedding_on_route(route, payload)
                self._set_route(route, reason="recognition-success")
                return result
            except Exception as exc:
                errors.append(f"{route.mode}: {exc}")

        raise RuntimeError("; ".join(errors) if errors else "All routes failed.")

    def send_embedding(self, embedding: list[float]) -> dict[str, Any]:
        payload = self._build_recognition_payload(embedding)
        try:
            result = self._send_with_failover(payload)
            flushed = self.flush_queue(max_items=self.cfg.queue_flush_batch)
            if flushed > 0:
                print(f"[queue] flushed {flushed} queued embeddings")
            return result
        except Exception as exc:
            self.queue.enqueue(payload)
            queued = self.queue.size()
            print(f"[queue] queued embedding (size={queued}) reason={exc}")
            return {"matched": False, "queued": True, "queue_size": queued}

    def flush_queue(self, max_items: int | None = None) -> int:
        batch_size = max_items if max_items is not None else self.cfg.queue_flush_batch
        pending = self.queue.fetch_batch(batch_size)
        if not pending:
            return 0

        ack_ids: list[int] = []
        flushed = 0
        for item in pending:
            try:
                self._send_with_failover(item.payload)
            except Exception:
                break
            ack_ids.append(item.id)
            flushed += 1

        self.queue.acknowledge(ack_ids)
        return flushed

    def heartbeat_loop(self) -> None:
        while True:
            try:
                route = self._choose_route()
                self._set_route(route, reason="heartbeat-loop")
                self._heartbeat_on_route(self.route)
                flushed = self.flush_queue(max_items=self.cfg.queue_flush_batch)
                if flushed > 0:
                    print(f"[queue] background flushed {flushed}")
            except Exception as exc:
                print(f"[heartbeat] failed: {exc}")
            time.sleep(self.cfg.heartbeat_seconds)

    def ws_target(self) -> tuple[str, str]:
        route = self._choose_route()
        self._set_route(route, reason="ws-target")
        try:
            token = self._login_on_route(self.route)
        except Exception:
            return (self.route.ws_url, "")
        return (self.route.ws_url, token)
