"""Тесты для FastAPI endpoints."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


@pytest.fixture
def client():
    """TestClient с замоканной моделью."""
    # Mock model loading
    mock_models = {"xgb": MagicMock()}
    mock_models["xgb"].predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch("api.main.load_model") as mock_load:
        mock_load.return_value = (mock_models, {}, {})
        from fastapi.testclient import TestClient
        from api.main import app, startup
        startup()
        yield TestClient(app)


VALID_HEADERS = {"X-API-Key": "dev-key-123"}

SAMPLE_APPLICANT = {
    "oblast": "Алматинская",
    "district": "Район1",
    "direction": "Скотоводство",
    "subsidy_name": "Приобретение племенного поголовья",
    "normative": 50000,
    "amount": 500000,
    "month": 3,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_score_without_api_key(client):
    resp = client.post("/score", json=SAMPLE_APPLICANT)
    assert resp.status_code == 401


def test_score_with_wrong_key(client):
    resp = client.post("/score", json=SAMPLE_APPLICANT,
                       headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_batch_without_api_key(client):
    resp = client.post("/score/batch", json=[SAMPLE_APPLICANT])
    assert resp.status_code == 401
