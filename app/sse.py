"""Server-Sent Events transport for the pluggable LLM token stream.

The encoder turns a stream of opaque token strings into the wire format defined
by the EventSource spec: lines prefixed with ``data:`` and terminated by a
blank line. A final ``event: done`` frame lets clients distinguish a clean
end-of-stream from a transport drop.

The token source itself is injected as a ``TokenStreamer`` so the same
transport works against the deterministic mock generator (tests, local dev)
and against a real LLM SDK (production) without touching this module.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.llm import LLMStreamError, TokenStreamer, get_token_streamer


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


async def sse_token_stream(
    streamer: TokenStreamer, prompt: str, delay_seconds: float
) -> AsyncIterator[str]:
    """Wrap a token stream in SSE frames + a terminal ``done`` / ``error`` event.

    The inner iterator is explicitly closed in ``finally`` so an early consumer
    drop (browser navigation, proxy timeout, CDN tear-down) propagates upstream
    immediately rather than waiting for the async-generator finalizer.
    """
    iterator = streamer.stream(prompt, delay_seconds=delay_seconds)
    try:
        async for token in iterator:
            yield format_sse_event(token, event="token")
    except LLMStreamError as exc:
        yield format_sse_event(str(exc), event="error")
        return
    finally:
        await iterator.aclose()
    yield format_sse_event("[DONE]", event="done")


router = APIRouter()


@router.get("/stream/sse")
async def stream_sse(
    prompt: str = Query("hello", min_length=0, max_length=2048),
    delay_seconds: float = Query(0.02, ge=0.0, le=1.0),
    streamer: TokenStreamer = Depends(get_token_streamer),
) -> StreamingResponse:
    return StreamingResponse(
        sse_token_stream(streamer, prompt, delay_seconds),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_STREAMING_HEADERS,
    )
