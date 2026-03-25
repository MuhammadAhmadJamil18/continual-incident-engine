"""Run the ASGI app with Uvicorn (development server)."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run(
        "incident_memory_engine.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
