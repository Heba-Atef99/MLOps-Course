from pathlib import Path

from app.model import IRIS_CLASS_NAMES, load_model, predict

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.skops"


def test_predict_returns_valid_class():
    model = load_model(MODEL_PATH)
    result = predict(model, [5.1, 3.5, 1.4, 0.2])
    assert result["class_name"] in IRIS_CLASS_NAMES


def test_predict_probabilities_sum_to_one():
    model = load_model(MODEL_PATH)
    result = predict(model, [5.1, 3.5, 1.4, 0.2])
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-3


def test_predict_output_structure():
    model = load_model(MODEL_PATH)
    result = predict(model, [6.7, 3.0, 5.2, 2.3])
    assert "class_name" in result
    assert "class_index" in result
    assert "probabilities" in result
    assert isinstance(result["class_index"], int)
    assert 0 <= result["class_index"] <= 2
    assert set(result["probabilities"].keys()) == set(IRIS_CLASS_NAMES)
