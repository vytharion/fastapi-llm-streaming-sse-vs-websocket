"""Decision matrix + production checklist for SSE vs WebSocket streaming.

Steps 1-7 built two parallel transports — SSE for plain HTTP token streams,
WebSocket for full-duplex sessions with mid-stream cancellation — and stress-
tested them under the same load shapes. The remaining engineering question is
*which one to pick for a new project*, and *what must be true before it ships*.

This module collapses both questions into runnable data:

- ``recommend_transport`` walks a small ranked rule set over a
  ``StreamingRequirements`` object and returns the chosen transport plus the
  reasons that drove the choice.
- ``production_checklist`` returns the list of go-live items that apply to a
  given transport. ``missing_checklist_items`` diffs that list against the IDs
  the operator says are already signed off, so a release gate can answer
  "are we done" without humans reading the matrix.

Both pieces are exposed over an HTTP router so the article, a runbook, or a
CI smoke test can pull them from the running app rather than copy-pasting
from a wiki that drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from fastapi import APIRouter, HTTPException


class Transport(str, Enum):
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass(frozen=True)
class StreamingRequirements:
    """Binary requirements the team has agreed on for the new stream."""

    needs_client_to_server_messages: bool = False
    needs_midstream_cancel: bool = False
    needs_binary_frames: bool = False
    needs_auto_reconnect: bool = False
    browser_only_consumers: bool = False
    behind_legacy_http_proxy: bool = False


@dataclass(frozen=True)
class Recommendation:
    transport: Transport
    reasons: tuple[str, ...]


# Any one of these requirements rules SSE out, so the rule set short-circuits
# on the first match and emits one reason per matching requirement.
_WS_FORCING_RULES: tuple[tuple[str, str], ...] = (
    (
        "needs_client_to_server_messages",
        "Bidirectional messaging requires WebSocket; SSE is server-to-client only.",
    ),
    (
        "needs_midstream_cancel",
        "Mid-stream cancel from the client needs a back-channel SSE doesn't provide.",
    ),
    (
        "needs_binary_frames",
        "Binary frames need a non-text transport; SSE only carries UTF-8 text.",
    ),
)

# Among streams where SSE is viable, these requirements actively favour it.
_SSE_PREFERRING_RULES: tuple[tuple[str, str], ...] = (
    (
        "needs_auto_reconnect",
        "EventSource auto-reconnects with Last-Event-ID for free.",
    ),
    (
        "browser_only_consumers",
        "EventSource is a browser-native API; no client library needed.",
    ),
    (
        "behind_legacy_http_proxy",
        "SSE rides plain HTTP/1.1 chunked responses, no Upgrade handshake.",
    ),
)

_DEFAULT_SSE_REASON: tuple[str, ...] = (
    "Default to SSE for one-way LLM token streaming: simpler ops, plays nicely with proxies, no Upgrade handshake.",
)


def _match_rules(
    reqs: StreamingRequirements,
    rules: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    return tuple(reason for attr, reason in rules if getattr(reqs, attr))


def recommend_transport(reqs: StreamingRequirements) -> Recommendation:
    """Pick a transport and explain why, given a set of requirements."""
    ws_reasons = _match_rules(reqs, _WS_FORCING_RULES)
    if ws_reasons:
        return Recommendation(Transport.WEBSOCKET, ws_reasons)
    sse_reasons = _match_rules(reqs, _SSE_PREFERRING_RULES)
    if sse_reasons:
        return Recommendation(Transport.SSE, sse_reasons)
    return Recommendation(Transport.SSE, _DEFAULT_SSE_REASON)


class ChecklistCategory(str, Enum):
    PROXY = "proxy"
    BACKPRESSURE = "backpressure"
    CANCELLATION = "cancellation"
    OBSERVABILITY = "observability"
    SECURITY = "security"
    RELIABILITY = "reliability"


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    category: ChecklistCategory
    description: str
    applies_to: frozenset[Transport]


_BOTH_TRANSPORTS: frozenset[Transport] = frozenset(
    {Transport.SSE, Transport.WEBSOCKET}
)
_SSE_TRANSPORT: frozenset[Transport] = frozenset({Transport.SSE})
_WS_TRANSPORT: frozenset[Transport] = frozenset({Transport.WEBSOCKET})


ALL_CHECKS: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        id="proxy.idle-timeout",
        category=ChecklistCategory.PROXY,
        description="Proxy / CDN idle timeout exceeds slow-stream length, or app sends keepalive frames.",
        applies_to=_BOTH_TRANSPORTS,
    ),
    ChecklistItem(
        id="backpressure.upstream-cancel",
        category=ChecklistCategory.BACKPRESSURE,
        description="Upstream LLM iterator is closed in `finally` when the consumer drops.",
        applies_to=_BOTH_TRANSPORTS,
    ),
    ChecklistItem(
        id="observability.per-stream-metrics",
        category=ChecklistCategory.OBSERVABILITY,
        description="Emit per-stream metrics: start, tokens, terminal outcome, duration.",
        applies_to=_BOTH_TRANSPORTS,
    ),
    ChecklistItem(
        id="security.auth-on-connect",
        category=ChecklistCategory.SECURITY,
        description="Authenticate the caller at connect time; reject anonymous streams.",
        applies_to=_BOTH_TRANSPORTS,
    ),
    ChecklistItem(
        id="security.rate-limit",
        category=ChecklistCategory.SECURITY,
        description="Per-user rate limit on concurrent streams to cap LLM spend.",
        applies_to=_BOTH_TRANSPORTS,
    ),
    ChecklistItem(
        id="proxy.x-accel-buffering",
        category=ChecklistCategory.PROXY,
        description="Response sets `X-Accel-Buffering: no` so nginx doesn't buffer the stream.",
        applies_to=_SSE_TRANSPORT,
    ),
    ChecklistItem(
        id="proxy.cache-control",
        category=ChecklistCategory.PROXY,
        description="Response sets `Cache-Control: no-cache` so CDNs don't cache the stream.",
        applies_to=_SSE_TRANSPORT,
    ),
    ChecklistItem(
        id="reliability.event-id-resume",
        category=ChecklistCategory.RELIABILITY,
        description="Emit a stable `id:` per frame so EventSource clients can resume via Last-Event-ID.",
        applies_to=_SSE_TRANSPORT,
    ),
    ChecklistItem(
        id="cancellation.cancel-frame",
        category=ChecklistCategory.CANCELLATION,
        description="Server honours a client `cancel` control frame mid-stream.",
        applies_to=_WS_TRANSPORT,
    ),
    ChecklistItem(
        id="reliability.heartbeat",
        category=ChecklistCategory.RELIABILITY,
        description="App-level heartbeat (ping / pong frame) detects dead peers within N seconds.",
        applies_to=_WS_TRANSPORT,
    ),
    ChecklistItem(
        id="reliability.normal-closure",
        category=ChecklistCategory.RELIABILITY,
        description="Close with code 1000 on clean shutdown; reserve non-1000 codes for errors.",
        applies_to=_WS_TRANSPORT,
    ),
    ChecklistItem(
        id="reliability.reconnect-backoff",
        category=ChecklistCategory.RELIABILITY,
        description="Client reconnects with exponential backoff + jitter to avoid thundering herd.",
        applies_to=_WS_TRANSPORT,
    ),
)


def production_checklist(transport: Transport) -> tuple[ChecklistItem, ...]:
    return tuple(item for item in ALL_CHECKS if transport in item.applies_to)


def missing_checklist_items(
    transport: Transport, completed_ids: Iterable[str]
) -> tuple[ChecklistItem, ...]:
    completed = set(completed_ids)
    full = production_checklist(transport)
    return tuple(item for item in full if item.id not in completed)


def _serialize_checklist_item(item: ChecklistItem) -> dict[str, object]:
    return {
        "id": item.id,
        "category": item.category.value,
        "description": item.description,
        "applies_to": sorted(t.value for t in item.applies_to),
    }


def _coerce_transport(raw: str) -> Transport:
    try:
        return Transport(raw)
    except ValueError as exc:
        valid = [t.value for t in Transport]
        raise HTTPException(
            status_code=422,
            detail=f"unknown transport {raw!r}; choose one of {valid}",
        ) from exc


router = APIRouter(prefix="/guidance", tags=["guidance"])


@router.get("/decision")
async def decision_endpoint(
    needs_client_to_server_messages: bool = False,
    needs_midstream_cancel: bool = False,
    needs_binary_frames: bool = False,
    needs_auto_reconnect: bool = False,
    browser_only_consumers: bool = False,
    behind_legacy_http_proxy: bool = False,
) -> dict[str, object]:
    reqs = StreamingRequirements(
        needs_client_to_server_messages=needs_client_to_server_messages,
        needs_midstream_cancel=needs_midstream_cancel,
        needs_binary_frames=needs_binary_frames,
        needs_auto_reconnect=needs_auto_reconnect,
        browser_only_consumers=browser_only_consumers,
        behind_legacy_http_proxy=behind_legacy_http_proxy,
    )
    recommendation = recommend_transport(reqs)
    return {
        "transport": recommendation.transport.value,
        "reasons": list(recommendation.reasons),
    }


@router.get("/checklist/{transport}")
async def checklist_endpoint(transport: str) -> dict[str, object]:
    transport_enum = _coerce_transport(transport)
    items = production_checklist(transport_enum)
    return {
        "transport": transport_enum.value,
        "items": [_serialize_checklist_item(item) for item in items],
    }
