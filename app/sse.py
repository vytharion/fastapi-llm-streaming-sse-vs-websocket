"""Server-Sent Events transport for the mock LLM token stream.

The encoder turns a stream of opaque token strings into the wire format defined
by the EventSource spec: lines prefixed with ``data:`` and terminated by a
blank line. A final ``event: done`` frame lets clients distinguish a clean
end-of-stream from a transport drop.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.llm_mock import stream_tokens


SSE_MEDIA_TYPE = "text/event-stream"

SSE_STREAMING_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable buffering on nginx / common reverse proxies so tokens flush.
    "X-Accel-Buffering": "no",
}


def format_sse_event(data: str, event: str | None = None) -> str:
    """Encode a single SSE frame.

    Multi-line payloads are split into one ``data:`` line per source line, as
    required by the EventSource spec.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    for chunk in data.split("\n"):
        lines.append(f"data: {chunk}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


async def sse_token_stream(prompt: str, delay_seconds: float) -> AsyncIterator[str]:
    """Wrap the mock token generator in SSE frames + a terminal done event."""
    async for token in stream_tokens(prompt, delay_seconds=delay_seconds):
        yield format_sse_event(token, event="token")
    yield format_sse_event("[DONE]", event="done")


router = APIRouter()


@router.get("/stream/sse")
async def stream_sse(
    prompt: str = Query("hello", min_length=0, max_length=2048),
    delay_seconds: float = Query(0.02, ge=0.0, le=1.0),
) -> StreamingResponse:
    return StreamingResponse(
        sse_token_stream(prompt, delay_seconds),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_STREAMING_HEADERS,
    )
