from litestar.testing import TestClient

from app.main import app

SAMPLE_REQUEST = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000,
}


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
        response = client.post("/predict", json=SAMPLE_REQUEST)
        assert response.status_code == 201
        data = response.json()
        assert "loan_approved" in data
        assert "approval_probability" in data


def test_predict_invalid_input():
    with TestClient(app=app) as client:
        response = client.post(
            "/predict",
            json={"cibil_score": "not_a_number"},
        )
        assert response.status_code == 400
