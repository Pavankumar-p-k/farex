from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


settings = get_settings()


def _normalize_database_url(database_url: str) -> str:
    # Render provides PostgreSQL URLs without an explicit driver.
    if database_url.startswith("postgresql://"):
        return f"postgresql+psycopg://{database_url[len('postgresql://'):]}"
    if database_url.startswith("postgres://"):
        return f"postgresql+psycopg://{database_url[len('postgres://'):]}"
    return database_url


engine = create_engine(_normalize_database_url(settings.database_url), future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

