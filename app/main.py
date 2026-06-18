"""FastAPI entrypoint.

Step 1 wired up a health check, step 2 mounted the SSE streaming router,
step 3 added the static EventSource browser client, and step 4 mounts the
WebSocket streaming router. The WebSocket browser client follows in a later
step.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.client import router as client_router
from app.sse import router as sse_router
from app.ws import router as ws_router


app = FastAPI(
    title="FastAPI LLM streaming demo",
    version=__version__,
    description="Companion app for the SSE vs WebSocket streaming tutorial.",
)

app.include_router(sse_router)
app.include_router(ws_router)
app.include_router(client_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
