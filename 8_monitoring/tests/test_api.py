from litestar.testing import TestClient

from app.main import app


def test_home_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data


def test_health_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_predict_endpoint():
    with TestClient(app=app) as client:
        response = client.post(
            "/predict",
            json={
                "hour_of_day": 14,
                "device_type": 1,
                "ad_position": 2,
                "user_age": 28,
                "session_duration_sec": 450.0,
                "page_views": 12,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "prediction_id" in data
        assert "predicted_click" in data
        assert "confidence" in data


def test_predict_then_feedback():
    with TestClient(app=app) as client:
        pred_resp = client.post(
            "/predict",
            json={
                "hour_of_day": 14,
                "device_type": 1,
                "ad_position": 2,
                "user_age": 28,
                "session_duration_sec": 450.0,
                "page_views": 12,
            },
        )
        assert pred_resp.status_code == 201
        prediction_id = pred_resp.json()["prediction_id"]

        fb_resp = client.post(
            "/feedback",
            json={
                "prediction_id": prediction_id,
                "clicked": True,
            },
        )
        assert fb_resp.status_code == 201
        fb_data = fb_resp.json()
        assert fb_data["prediction_id"] == prediction_id
        assert "correct" in fb_data
        assert "actual_click" in fb_data


def test_feedback_unknown_prediction():
    with TestClient(app=app) as client:
        response = client.post(
            "/feedback",
            json={
                "prediction_id": "nonexistent-id",
                "clicked": True,
            },
        )
        assert response.status_code == 404


def test_predict_invalid_input():
    with TestClient(app=app) as client:
        response = client.post(
            "/predict",
            json={"hour_of_day": "not_a_number"},
        )
        assert response.status_code == 400
