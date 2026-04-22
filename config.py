"""Application configuration loaded from environment variables.

Why this file exists:
- Centralize runtime settings for local/dev/prod deployments.
- Remove hardcoded secrets from source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency during runtime bootstrap
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    """Fetch an environment variable with optional default."""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_env_int(name: str, default: int) -> int:
    """Fetch an integer environment variable with validation."""
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw}") from exc


def _get_env_float(name: str, default: float) -> float:
    """Fetch a float environment variable with validation."""
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings used by database, liveness, and service layers."""

    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_min_connections: int
    db_max_connections: int
    model_name: str
    model_detector_backend: str
    liveness_timeout_seconds: int
    min_passive_for_match: float


def load_settings() -> Settings:
    """Load validated settings from environment variables."""
    return Settings(
        db_name=_get_env("DB_NAME", "face_db"),
        db_user=_get_env("DB_USER", "postgres"),
        db_password=_get_env("DB_PASSWORD", ""),
        db_host=_get_env("DB_HOST", "localhost"),
        db_port=_get_env_int("DB_PORT", 5432),
        db_min_connections=_get_env_int("DB_MIN_CONNECTIONS", 1),
        db_max_connections=_get_env_int("DB_MAX_CONNECTIONS", 5),
        model_name=_get_env("MODEL_NAME", "ArcFace"),
        model_detector_backend=_get_env("MODEL_DETECTOR_BACKEND", "opencv"),
        liveness_timeout_seconds=_get_env_int("LIVENESS_TIMEOUT_SECONDS", 18),
        min_passive_for_match=_get_env_float("MIN_PASSIVE_FOR_MATCH", 0.30),
    )


settings = load_settings()
