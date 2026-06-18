"""FastAPI entrypoint.

Step 1 wired up a health check, step 2 mounted the SSE streaming router,
step 3 added the static EventSource browser client, step 4 added the
WebSocket streaming router, step 5 mounted a second static page — a
WebSocket browser client with reconnect logic and an application-level
heartbeat — step 6 swapped the deterministic mock generator for a
pluggable ``TokenStreamer`` so both transports can sit in front of a real
LLM SDK selected at startup via environment variables, step 7 added
stress / robustness tests covering backpressure, dropped clients, proxy
header durability, and multiplexing, and step 8 publishes the decision
matrix + production checklist as runnable data behind a ``/guidance``
router so the article, runbooks, and CI gates can pull them from the
running app instead of copying from a wiki that drifts.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.client import router as client_router
from app.guidance import router as guidance_router
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
app.include_router(guidance_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
