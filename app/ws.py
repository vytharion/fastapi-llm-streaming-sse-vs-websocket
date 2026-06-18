"""WebSocket transport for the mock LLM token stream.

Unlike SSE, WebSocket gives us a full-duplex channel so the client can ask the
server to stop mid-stream without opening a second HTTP request. Two cooperating
tasks run for the lifetime of the connection:

- a *streamer* that pushes JSON-encoded ``token`` frames from the mock
  generator,
- a *canceller* that blocks on ``receive_text`` until either a cancel control
  message arrives or the client disconnects.

Whichever finishes first wins; the loser is cancelled, a terminal ``done`` or
``cancelled`` frame is emitted when possible, and the socket is closed with the
1000 normal-closure code so proxies don't log the connection as an error.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.llm_mock import stream_tokens


WS_NORMAL_CLOSURE = 1000
CANCEL_ACTION = "cancel"

OUTCOME_DONE = "done"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_DISCONNECTED = "disconnected"


def encode_token_frame(token: str) -> str:
    return json.dumps({"type": "token", "data": token})


def encode_done_frame() -> str:
    return json.dumps({"type": OUTCOME_DONE, "data": "[DONE]"})


def encode_cancelled_frame() -> str:
    return json.dumps({"type": OUTCOME_CANCELLED, "data": "[CANCELLED]"})


def _try_parse_json(raw: str) -> object | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def is_cancel_message(raw: str) -> bool:
    """Accept either a bare ``cancel`` keyword or ``{"action": "cancel"}``."""
    stripped = raw.strip()
    if stripped.lower() == CANCEL_ACTION:
        return True
    parsed = _try_parse_json(stripped)
    if not isinstance(parsed, dict):
        return False
    return parsed.get("action") == CANCEL_ACTION


async def _push_tokens(
    websocket: WebSocket, prompt: str, delay_seconds: float
) -> str:
    """Stream tokens until generator drains or the peer drops the socket."""
    try:
        async for token in stream_tokens(prompt, delay_seconds=delay_seconds):
            await websocket.send_text(encode_token_frame(token))
    except WebSocketDisconnect:
        return OUTCOME_DISCONNECTED
    return OUTCOME_DONE


async def _await_cancel(websocket: WebSocket) -> str:
    """Block on incoming frames until a cancel arrives or the peer drops."""
    try:
        while True:
            raw = await websocket.receive_text()
            if is_cancel_message(raw):
                return OUTCOME_CANCELLED
    except WebSocketDisconnect:
        return OUTCOME_DISCONNECTED


async def _drain_pending(tasks: list[asyncio.Task[str]]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _safe_result(task: asyncio.Task[str]) -> str | None:
    if task.cancelled():
        return None
    if task.exception() is not None:
        return None
    return task.result()


async def run_websocket_stream(
    websocket: WebSocket,
    prompt: str,
    delay_seconds: float,
) -> str:
    """Drive both halves of the WS until one terminates. Return outcome label."""
    streamer = asyncio.create_task(_push_tokens(websocket, prompt, delay_seconds))
    canceller = asyncio.create_task(_await_cancel(websocket))

    done, _ = await asyncio.wait(
        {streamer, canceller},
        return_when=asyncio.FIRST_COMPLETED,
    )
    await _drain_pending([streamer, canceller])

    canceller_result = _safe_result(canceller) if canceller in done else None
    if canceller_result == OUTCOME_CANCELLED:
        return OUTCOME_CANCELLED
    streamer_result = _safe_result(streamer) if streamer in done else None
    if streamer_result == OUTCOME_DONE:
        return OUTCOME_DONE
    return OUTCOME_DISCONNECTED


async def _send_terminal_frame(websocket: WebSocket, payload: str) -> None:
    if websocket.application_state != WebSocketState.CONNECTED:
        return
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    try:
        await websocket.send_text(payload)
    except (WebSocketDisconnect, RuntimeError):
        return


async def _close_quietly(websocket: WebSocket) -> None:
    if websocket.application_state == WebSocketState.DISCONNECTED:
        return
    try:
        await websocket.close(code=WS_NORMAL_CLOSURE)
    except (WebSocketDisconnect, RuntimeError):
        return


async def _emit_outcome(websocket: WebSocket, outcome: str) -> None:
    if outcome == OUTCOME_DISCONNECTED:
        return
    if outcome == OUTCOME_CANCELLED:
        await _send_terminal_frame(websocket, encode_cancelled_frame())
        return
    await _send_terminal_frame(websocket, encode_done_frame())


router = APIRouter()


@router.websocket("/stream/ws")
async def stream_ws(
    websocket: WebSocket,
    prompt: str = Query("hello", min_length=0, max_length=2048),
    delay_seconds: float = Query(0.02, ge=0.0, le=1.0),
) -> None:
    await websocket.accept()
    outcome = await run_websocket_stream(websocket, prompt, delay_seconds)
    await _emit_outcome(websocket, outcome)
    await _close_quietly(websocket)
