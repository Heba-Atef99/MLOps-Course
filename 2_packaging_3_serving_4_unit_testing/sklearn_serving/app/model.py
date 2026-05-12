from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.pipeline import Pipeline

PRED_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def load_model(path: Path) -> Pipeline:
    unknown_types = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=unknown_types)


def predict(model: Pipeline, features: dict) -> dict:
    df = pd.DataFrame([features])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "loan_approved": prediction == 1,
        "approval_probability": round(probability, 4),
    }
