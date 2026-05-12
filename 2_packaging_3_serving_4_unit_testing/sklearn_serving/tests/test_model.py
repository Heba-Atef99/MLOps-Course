from pathlib import Path

from app.model import load_model, predict

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "loan_model.skops"

SAMPLE_APPROVED = {
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

SAMPLE_REJECTED = {
    "no_of_dependents": 0,
    "education": "Not Graduate",
    "self_employed": "Yes",
    "income_annum": 4100000,
    "loan_amount": 12200000,
    "loan_term": 8,
    "cibil_score": 417,
    "residential_assets_value": 2700000,
    "commercial_assets_value": 2200000,
    "luxury_assets_value": 8800000,
    "bank_asset_value": 3300000,
}


def test_predict_returns_boolean():
    model = load_model(MODEL_PATH)
    result = predict(model, SAMPLE_APPROVED)
    assert isinstance(result["loan_approved"], bool)


def test_predict_probability_range():
    model = load_model(MODEL_PATH)
    result = predict(model, SAMPLE_APPROVED)
    assert 0.0 <= result["approval_probability"] <= 1.0


def test_predict_output_structure():
    model = load_model(MODEL_PATH)
    result = predict(model, SAMPLE_REJECTED)
    assert "loan_approved" in result
    assert "approval_probability" in result
