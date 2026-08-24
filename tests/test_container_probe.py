"""Regression test for the container-health probe used by CI."""

from __future__ import annotations

from scripts import check_container_parity as probe


def test_wait_for_health_retries_connection_reset(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_request(url: str, *, payload=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionResetError("container socket opened before application readiness")
        return {"status": "ok", "model_loaded": True}

    monkeypatch.setattr(probe, "_json_request", fake_request)

    health = probe.wait_for_health(
        "http://127.0.0.1:8000",
        attempts=3,
        delay=0.0,
    )

    assert health == {"status": "ok", "model_loaded": True}
    assert calls["count"] == 3
