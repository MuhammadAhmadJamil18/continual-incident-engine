from __future__ import annotations

import os

from fastapi import HTTPException, Request

from incident_memory_engine import __version__
from incident_memory_engine.api.responses import envelope_error


def parse_api_keys() -> set[str]:
    raw = os.environ.get("IME_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def extract_api_key(request: Request) -> str | None:
    h = request.headers.get("X-API-Key")
    if h:
        return h.strip()
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def require_api_key_if_configured(request: Request) -> None:
    keys = parse_api_keys()
    if not keys:
        return
    token = extract_api_key(request)
    if not token or token not in keys:
        rid = getattr(request.state, "request_id", "") or ""
        raise HTTPException(
            status_code=401,
            detail=envelope_error(
                "unauthorized",
                "Invalid or missing API key",
                request_id=rid,
                version=__version__,
            ),
        )


def rate_limit_per_minute() -> int:
    try:
        return max(1, int(os.environ.get("IME_RATE_LIMIT_PER_MINUTE", "120")))
    except ValueError:
        return 120


def max_ingest_rows() -> int:
    try:
        return max(1, int(os.environ.get("IME_MAX_INGEST_ROWS", "20000")))
    except ValueError:
        return 20000


def max_upload_bytes() -> int:
    try:
        return max(1024, int(os.environ.get("IME_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))))
    except ValueError:
        return 20 * 1024 * 1024
