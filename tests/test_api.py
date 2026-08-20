"""FastAPI contract via Starlette TestClient."""

from __future__ import annotations


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_contains_versions(client, train_result):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == train_result.metadata["schema_version"]
    assert body["model_version"] == train_result.metadata["model_version"]
    assert body["feature_order"] == train_result.metadata["feature_order"]
    assert body["seed"] == 2026


def test_predict_and_predict_proba(client):
    payload = {"age": 55.0, "income": 90000.0, "credit_score": 0.8, "segment": "C"}
    pred = client.post("/predict", json=payload)
    proba = client.post("/predict-proba", json=payload)
    assert pred.status_code == 200
    assert proba.status_code == 200
    assert pred.json()["prediction"] in (0, 1)
    p1 = proba.json()["probability_1"]
    assert 0.0 <= p1 <= 1.0
    assert proba.json()["prediction"] == int(p1 >= 0.5)


def test_invalid_requests_are_422(client):
    base = {"age": 40.0, "income": 35000.0, "credit_score": 0.6, "segment": "B"}
    missing = {k: v for k, v in base.items() if k != "income"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**base, "age": "x"}).status_code == 422
    assert client.post("/predict", json={**base, "age": 9.0}).status_code == 422
    assert client.post("/predict", json={**base, "segment": "Z"}).status_code == 422
    assert client.post("/predict", json={**base, "extra": 1}).status_code == 422
    assert client.post("/predict-proba", json={**base, "credit_score": 1.5}).status_code == 422


def test_health_without_model():
    from fastapi.testclient import TestClient

    from mlserv.api.app import create_app

    app = create_app(bundle=None)
    app.state.bundle = None
    local = TestClient(app)
    health = local.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert local.get("/model-info").status_code == 503
    payload = {"age": 40.0, "income": 35000.0, "credit_score": 0.6, "segment": "B"}
    assert local.post("/predict", json=payload).status_code == 503
