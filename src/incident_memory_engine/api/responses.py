from __future__ import annotations

import uuid
from typing import Any


def new_request_id() -> str:
    return str(uuid.uuid4())


def envelope_ok(
    data: Any,
    *,
    request_id: str,
    version: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data,
        "error": None,
        "meta": {"request_id": request_id, "version": version},
    }


def envelope_error(
    code: str,
    message: str,
    *,
    details: Any = None,
    request_id: str = "",
    version: str = "",
) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message, "details": details},
        "meta": {"request_id": request_id, "version": version},
    }
