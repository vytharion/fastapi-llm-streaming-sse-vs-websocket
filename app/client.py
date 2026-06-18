"""Static browser client that consumes the SSE stream via EventSource.

The HTML lives next to this module so it ships with the package and is reachable
without an extra deployment step. The router exposes the page at ``/`` and the
raw file under ``/client/index.html`` for tooling that wants a stable filename.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


CLIENT_HTML_PATH: Path = Path(__file__).parent / "static" / "index.html"


def load_client_html() -> str:
    """Read the bundled client HTML from disk.

    Kept as a function so tests can call it without spinning up the HTTP stack
    and so future caching can slot in without touching the route handler.
    """
    return CLIENT_HTML_PATH.read_text(encoding="utf-8")


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
