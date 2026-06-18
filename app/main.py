"""FastAPI entrypoint.

Step 1 only wires up a health check; the SSE and WebSocket endpoints arrive
in the following steps.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__


app = FastAPI(
    title="FastAPI LLM streaming demo",
    version=__version__,
    description="Companion app for the SSE vs WebSocket streaming tutorial.",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
