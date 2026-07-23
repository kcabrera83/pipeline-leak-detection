import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "pipeline-leak-detection"


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code in (200, 503)


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_valid(client):
    response = client.post("/api/predict", json={})
    assert response.status_code in (200, 400, 503)


def test_predict_with_params(client):
    payload = {
        "pipeline_type": "gas",
        "pipeline_length_km": 100.0,
        "pipeline_diameter_mm": 300.0,
        "pressure_upstream_mpa": 6.0,
        "pressure_downstream_mpa": 5.5,
        "flow_rate_m3h": 300.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (200, 400, 503)


def test_batch_valid(client):
    readings = [
        {"pipeline_type": "crude_oil", "pressure_upstream_mpa": 5.0},
        {"pipeline_type": "gas", "pressure_upstream_mpa": 4.0},
    ]
    response = client.post("/api/batch", json={"readings": readings})
    assert response.status_code in (200, 400, 503)


def test_batch_empty(client):
    response = client.post("/api/batch", json={"readings": []})
    assert response.status_code in (200, 400, 503)
