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
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["class_name"] in ["setosa", "versicolor", "virginica"]
        assert "class_index" in data
        assert "probabilities" in data


def test_predict_invalid_input():
    with TestClient(app=app) as client:
        response = client.post(
            "/predict",
            json={"sepal_length": "not_a_number"},
        )
        assert response.status_code == 400
