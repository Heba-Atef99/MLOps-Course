from pathlib import Path

from app.model import load_model, predict

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "ctr_model.skops"


def test_predict_returns_boolean_click():
    model = load_model(MODEL_PATH)
    result = predict(model, [14, 1, 2, 28, 450.0, 12])
    assert isinstance(result["predicted_click"], bool)


def test_predict_confidence_range():
    model = load_model(MODEL_PATH)
    result = predict(model, [14, 1, 2, 28, 450.0, 12])
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_output_structure():
    model = load_model(MODEL_PATH)
    result = predict(model, [3, 0, 5, 60, 1500.0, 2])
    assert "predicted_click" in result
    assert "confidence" in result
    assert "click_probability" in result
    assert 0.0 <= result["click_probability"] <= 1.0
