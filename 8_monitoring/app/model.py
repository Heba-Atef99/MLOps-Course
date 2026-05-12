from pathlib import Path

import numpy as np
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    "hour_of_day",
    "device_type",
    "ad_position",
    "user_age",
    "session_duration_sec",
    "page_views",
]


def load_model(path: Path) -> RandomForestClassifier:
    unknown_types = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=unknown_types)


def predict(model: RandomForestClassifier, features: list[float]) -> dict:
    X = np.array(features).reshape(1, -1)
    predicted_click = bool(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))

    return {
        "predicted_click": predicted_click,
        "confidence": round(confidence, 4),
        "click_probability": round(float(probabilities[1]), 4),
    }
