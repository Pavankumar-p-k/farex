from __future__ import annotations

import secrets
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Distributed Face Platform"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/face_platform"
    local_server_id: str = "local-site-a"

    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60

    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = "ChangeMe123!"
    bootstrap_admin_role: str = "admin"

    embedding_cipher_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    match_threshold: float = 0.62
    attendance_cooldown_seconds: int = 15

    sync_enabled: bool = True
    sync_interval_seconds: int = 30
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_schema: str = "public"

    cors_origins_raw: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

