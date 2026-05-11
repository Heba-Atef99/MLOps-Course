from pathlib import Path

import numpy as np
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier

IRIS_CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_model(path: Path) -> RandomForestClassifier:
    unknown_types = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=unknown_types)


def predict(model: RandomForestClassifier, features: list[float]) -> dict:
    X = np.array(features).reshape(1, -1)
    class_index = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    return {
        "class_name": IRIS_CLASS_NAMES[class_index],
        "class_index": class_index,
        "probabilities": {
            name: round(float(prob), 4)
            for name, prob in zip(IRIS_CLASS_NAMES, probabilities, strict=True)
        },
    }
