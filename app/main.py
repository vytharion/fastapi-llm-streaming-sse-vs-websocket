"""FastAPI entrypoint.

Step 1 wired up a health check; step 2 mounts the SSE streaming router. The
WebSocket endpoint follows in a later step.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.sse import router as sse_router


app = FastAPI(
    title="FastAPI LLM streaming demo",
    version=__version__,
    description="Companion app for the SSE vs WebSocket streaming tutorial.",
)

app.include_router(sse_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
