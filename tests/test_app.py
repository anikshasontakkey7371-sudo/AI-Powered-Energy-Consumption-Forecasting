"""
Tests for Flask REST API and Web Dashboard.
"""

import json
from pathlib import Path
import pytest
from app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_dashboard_route(client):
    """Verify root / dashboard renders successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI-Powered Energy Consumption Forecasting System" in response.data
    assert b"24-Hour Ahead Hourly Forecast" in response.data


def test_health_check_endpoint(client):
    """Verify /health returns JSON with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "service" in data
    assert data["unit"] == "MWh"


def test_predict_endpoint_validation_errors(client):
    """Verify /api/predict returns 400 Bad Request on invalid payloads."""
    # Test non-JSON request
    res1 = client.post("/api/predict", data="not json")
    assert res1.status_code == 400

    # Test empty JSON
    res2 = client.post("/api/predict", json={})
    assert res2.status_code == 400
    assert b"Missing 'features' key" in res2.data


def test_get_forecast_endpoint(client):
    """Verify /api/forecast returns valid JSON with 24 forecast records."""
    response = client.get("/api/forecast")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "records" in data
    assert "forecast_horizon_hours" in data
    assert data["forecast_horizon_hours"] == 24
    assert len(data["records"]) == 24
    assert "Forecast_Energy" in data["records"][0]


def test_get_metrics_endpoint(client):
    """Verify /api/metrics returns valid JSON with comparative metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "metrics" in data
    assert len(data["metrics"]) >= 2
    model_names = [m["model_name"] for m in data["metrics"]]
    assert any("Baseline" in name for name in model_names)
    assert any("RandomForest" in name for name in model_names)


def test_predict_endpoint_success(client):
    """Verify /api/predict with valid features returns 200 and prediction."""
    valid_payload = {
        "features": {
            "hour": 14,
            "dayofweek": 2,
            "month": 6,
            "is_weekend": 0,
            "lag_1": 55.2,
            "lag_2": 54.1,
            "lag_3": 52.8,
            "lag_24": 53.9,
            "lag_48": 54.0,
            "lag_168": 56.1,
            "rolling_mean_24": 51.5,
            "rolling_std_24": 4.2,
            "rolling_mean_168": 50.8,
        }
    }
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0
    assert data["unit"] == "MWh"


def test_predict_endpoint_non_numeric_values(client):
    """Verify /api/predict returns 400 when feature values are non-numeric."""
    invalid_payload = {
        "features": {
            "hour": "not-a-number",
            "dayofweek": 2,
            "month": 6,
            "is_weekend": 0,
            "lag_1": 55.2,
            "lag_2": 54.1,
            "lag_3": 52.8,
            "lag_24": 53.9,
            "lag_48": 54.0,
            "lag_168": 56.1,
            "rolling_mean_24": 51.5,
            "rolling_std_24": 4.2,
            "rolling_mean_168": 50.8,
        }
    }
    response = client.post("/api/predict", json=invalid_payload)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_404_error_handler(client):
    """Verify custom 404 handler returns clean JSON error."""
    response = client.get("/api/unknown_endpoint")
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data["error"] == "Not Found"
