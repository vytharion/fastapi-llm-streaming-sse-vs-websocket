"""Static browser clients for the streaming demos.

Two HTML pages are bundled with the package so they ship without any extra
deployment step:

- ``index.html`` — the SSE / ``EventSource`` demo, mounted at ``/``.
- ``ws.html`` — the WebSocket demo with reconnect + heartbeat, mounted at
  ``/ws``.

Each page is also reachable under a stable ``/client/<file>`` alias for
tooling that wants to fetch the raw bundle by filename.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


STATIC_DIR: Path = Path(__file__).parent / "static"
CLIENT_HTML_PATH: Path = STATIC_DIR / "index.html"
WS_CLIENT_HTML_PATH: Path = STATIC_DIR / "ws.html"


def load_client_html() -> str:
    """Read the bundled SSE client HTML from disk."""
    return CLIENT_HTML_PATH.read_text(encoding="utf-8")


def load_ws_client_html() -> str:
    """Read the bundled WebSocket client HTML from disk."""
    return WS_CLIENT_HTML_PATH.read_text(encoding="utf-8")


router = APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(content=load_client_html())


@router.get(
    "/client/index.html",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def client_html() -> HTMLResponse:
    return HTMLResponse(content=load_client_html())


@router.get("/ws", response_class=HTMLResponse, include_in_schema=False)
async def ws_client_index() -> HTMLResponse:
    return HTMLResponse(content=load_ws_client_html())


@router.get(
    "/client/ws.html",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def ws_client_html() -> HTMLResponse:
    return HTMLResponse(content=load_ws_client_html())
