"""Train a loan approval prediction pipeline and serialize with skops."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import skops.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.preprocessing import (
    CustomLabelEncoder,
    DomainProcessing,
    DropColumns,
    LogTransforms,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DATA_PATH = Path(__file__).resolve().parent / "loan_approval_dataset.csv"

TARGET = "loan_status"
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
ASSET_COLUMNS = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
LABEL_ENCODE_MAP = {"education": ["Graduate"], "self_employed": ["Yes"]}
LOG_COLUMNS = ["income_annum", "loan_amount", "total_assets_value"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("domain", DomainProcessing(columns_to_sum=ASSET_COLUMNS)),
            ("drop", DropColumns(columns_to_drop=ASSET_COLUMNS)),
            ("label_encode", CustomLabelEncoder(encoding_map=LABEL_ENCODE_MAP)),
            ("log_transform", LogTransforms(columns=LOG_COLUMNS)),
            ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
        ]
    )


def train_and_save() -> None:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df[TARGET] = df[TARGET].apply(lambda x: 1 if x.strip() == "Approved" else 0)

    X = df[PRED_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    approval_rate = y.mean()
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Dataset approval rate: {approval_rate:.2%}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "loan_model.skops"
    sio.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")


if __name__ == "__main__":
    train_and_save()
