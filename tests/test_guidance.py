"""Tests for the decision matrix + production checklist (step 8).

These tests cover three angles of the ``guidance`` module:

- The pure decision function — does it default to SSE, does it short-circuit
  to WebSocket on any forcing requirement, does it surface every reason that
  drove the choice?
- The checklist data — are IDs unique, are SSE-only and WS-only items
  partitioned correctly, does ``missing_checklist_items`` actually subtract?
- The HTTP surface — do the ``/guidance/decision`` and
  ``/guidance/checklist/{transport}`` endpoints round-trip the same answers
  the pure functions return, and does the checklist endpoint reject unknown
  transports with a 422?
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.guidance import (
    ALL_CHECKS,
    Recommendation,
    StreamingRequirements,
    Transport,
    missing_checklist_items,
    production_checklist,
    recommend_transport,
)
from app.main import app


client = TestClient(app)


# === Decision matrix ===========================================================


def test_default_requirements_pick_sse_with_default_reason() -> None:
    rec = recommend_transport(StreamingRequirements())

    assert isinstance(rec, Recommendation)
    assert rec.transport is Transport.SSE
    assert len(rec.reasons) == 1
    assert "Default to SSE" in rec.reasons[0]


def test_needs_midstream_cancel_forces_websocket() -> None:
    rec = recommend_transport(
        StreamingRequirements(needs_midstream_cancel=True)
    )

    assert rec.transport is Transport.WEBSOCKET
    joined = " ".join(rec.reasons)
    assert "Mid-stream cancel" in joined


def test_needs_client_to_server_messages_forces_websocket() -> None:
    rec = recommend_transport(
        StreamingRequirements(needs_client_to_server_messages=True)
    )

    assert rec.transport is Transport.WEBSOCKET
    assert any("Bidirectional" in reason for reason in rec.reasons)


def test_binary_frames_force_websocket_even_when_sse_signals_present() -> None:
    rec = recommend_transport(
        StreamingRequirements(
            needs_binary_frames=True,
            browser_only_consumers=True,
            needs_auto_reconnect=True,
        )
    )

    assert rec.transport is Transport.WEBSOCKET
    assert all("EventSource" not in reason for reason in rec.reasons)


def test_browser_only_with_no_ws_pressure_picks_sse() -> None:
    rec = recommend_transport(
        StreamingRequirements(
            browser_only_consumers=True, needs_auto_reconnect=True
        )
    )

    assert rec.transport is Transport.SSE
    joined = " ".join(rec.reasons)
    assert "EventSource" in joined


def test_multiple_ws_signals_collect_all_reasons_in_declared_order() -> None:
    rec = recommend_transport(
        StreamingRequirements(
            needs_client_to_server_messages=True,
            needs_midstream_cancel=True,
            needs_binary_frames=True,
        )
    )

    assert rec.transport is Transport.WEBSOCKET
    assert len(rec.reasons) == 3
    assert "Bidirectional" in rec.reasons[0]
    assert "Mid-stream cancel" in rec.reasons[1]
    assert "Binary frames" in rec.reasons[2]


def test_recommendation_is_immutable() -> None:
    rec = recommend_transport(StreamingRequirements())

    with pytest.raises(Exception):
        rec.transport = Transport.WEBSOCKET  # type: ignore[misc]


# === Production checklist ======================================================


def test_every_check_applies_to_at_least_one_transport() -> None:
    for item in ALL_CHECKS:
        assert item.applies_to, item.id


def test_check_ids_are_unique() -> None:
    ids = [item.id for item in ALL_CHECKS]
    assert len(ids) == len(set(ids))


def test_sse_checklist_includes_x_accel_buffering_and_excludes_ws_only() -> None:
    items = production_checklist(Transport.SSE)
    ids = {item.id for item in items}

    assert "proxy.x-accel-buffering" in ids
    assert "proxy.cache-control" in ids
    assert "cancellation.cancel-frame" not in ids
    assert "reliability.heartbeat" not in ids


def test_ws_checklist_includes_cancel_and_heartbeat_excludes_sse_only() -> None:
    items = production_checklist(Transport.WEBSOCKET)
    ids = {item.id for item in items}

    assert "cancellation.cancel-frame" in ids
    assert "reliability.heartbeat" in ids
    assert "reliability.reconnect-backoff" in ids
    assert "proxy.x-accel-buffering" not in ids
    assert "reliability.event-id-resume" not in ids


def test_shared_checks_appear_in_both_transports() -> None:
    sse_ids = {item.id for item in production_checklist(Transport.SSE)}
    ws_ids = {item.id for item in production_checklist(Transport.WEBSOCKET)}
    shared = sse_ids & ws_ids

    assert "backpressure.upstream-cancel" in shared
    assert "security.auth-on-connect" in shared
    assert "security.rate-limit" in shared
    assert "observability.per-stream-metrics" in shared


def test_missing_items_filters_out_completed_ids() -> None:
    full = production_checklist(Transport.WEBSOCKET)
    completed = [item.id for item in full[:2]]

    missing = missing_checklist_items(Transport.WEBSOCKET, completed)

    missing_ids = {item.id for item in missing}
    for done_id in completed:
        assert done_id not in missing_ids
    assert len(missing) == len(full) - 2


def test_missing_items_with_empty_set_returns_full_list() -> None:
    full = production_checklist(Transport.SSE)
    missing = missing_checklist_items(Transport.SSE, [])

    assert missing == full


def test_missing_items_ignores_irrelevant_completed_ids() -> None:
    full = production_checklist(Transport.SSE)

    missing = missing_checklist_items(
        Transport.SSE,
        ["nonexistent.foo", "reliability.heartbeat"],  # WS-only ID
    )

    assert missing == full


# === HTTP endpoints ============================================================


def test_decision_endpoint_default_returns_sse() -> None:
    response = client.get("/guidance/decision")

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "sse"
    assert any("Default to SSE" in reason for reason in payload["reasons"])


def test_decision_endpoint_ws_pressure_returns_websocket() -> None:
    response = client.get(
        "/guidance/decision",
        params={"needs_midstream_cancel": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "websocket"
    assert any("Mid-stream cancel" in reason for reason in payload["reasons"])


def test_decision_endpoint_combines_multiple_ws_signals() -> None:
    response = client.get(
        "/guidance/decision",
        params={
            "needs_client_to_server_messages": "true",
            "needs_binary_frames": "true",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "websocket"
    assert len(payload["reasons"]) == 2


def test_checklist_endpoint_returns_sse_items() -> None:
    response = client.get("/guidance/checklist/sse")

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "sse"
    ids = {item["id"] for item in payload["items"]}
    assert "proxy.x-accel-buffering" in ids
    assert "backpressure.upstream-cancel" in ids


def test_checklist_endpoint_returns_ws_items_with_categories() -> None:
    response = client.get("/guidance/checklist/websocket")

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "websocket"
    categories = {item["category"] for item in payload["items"]}
    assert "cancellation" in categories
    assert "reliability" in categories
    assert "security" in categories


def test_checklist_endpoint_applies_to_field_lists_transports() -> None:
    response = client.get("/guidance/checklist/sse")

    payload = response.json()
    shared_item = next(
        item for item in payload["items"] if item["id"] == "security.auth-on-connect"
    )
    sse_only_item = next(
        item for item in payload["items"] if item["id"] == "proxy.x-accel-buffering"
    )

    assert shared_item["applies_to"] == ["sse", "websocket"]
    assert sse_only_item["applies_to"] == ["sse"]


def test_checklist_endpoint_rejects_unknown_transport() -> None:
    response = client.get("/guidance/checklist/quic")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "unknown transport" in detail


def test_guidance_endpoints_are_listed_in_openapi() -> None:
    response = client.get("/openapi.json")
    schema = response.json()

    paths = schema["paths"]
    assert "/guidance/decision" in paths
    assert "/guidance/checklist/{transport}" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
