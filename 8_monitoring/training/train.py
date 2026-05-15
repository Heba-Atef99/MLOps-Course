"""Train a click-through rate prediction model on synthetic data."""

from pathlib import Path

import numpy as np
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASELINE_PATH = DATA_DIR / "training_baseline.csv"

FEATURE_NAMES = [
    "hour_of_day",
    "device_type",
    "ad_position",
    "user_age",
    "session_duration_sec",
    "page_views",
]


def generate_synthetic_data(
    n_samples: int = 5000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    hour_of_day = rng.integers(0, 24, n_samples)
    device_type = rng.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    ad_position = rng.integers(1, 6, n_samples)
    user_age = rng.integers(18, 66, n_samples)
    session_duration = rng.uniform(0, 1800, n_samples).round(1)
    page_views = rng.integers(1, 51, n_samples)

    X = np.column_stack(
        [hour_of_day, device_type, ad_position, user_age, session_duration, page_views]
    )

    prob = np.full(n_samples, 0.15)
    prob -= (ad_position - 1) * 0.025
    prob += (device_type == 1) * 0.06
    prob += (device_type == 2) * 0.02
    is_peak = ((hour_of_day >= 10) & (hour_of_day <= 14)) | (
        (hour_of_day >= 19) & (hour_of_day <= 22)
    )
    prob += is_peak * 0.08
    prob += (page_views / 50) * 0.06
    prob += ((user_age >= 25) & (user_age <= 34)) * 0.04
    prob -= (session_duration > 1200) * 0.03
    prob = np.clip(prob, 0.02, 0.90)

    y = rng.binomial(1, prob)
    return X, y


def train_and_save() -> None:
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    ctr = y.mean()
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Dataset CTR: {ctr:.2%}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "ctr_model.skops"
    sio.dump(model, model_path)
    print(f"Model saved to {model_path}")

    DATA_DIR.mkdir(exist_ok=True)
    np.savetxt(
        BASELINE_PATH,
        X_train,
        delimiter=",",
        header=",".join(FEATURE_NAMES),
        comments="",
        fmt=["%d", "%d", "%d", "%d", "%.1f", "%d"],
    )
    print(f"Training baseline saved to {BASELINE_PATH}")


if __name__ == "__main__":
    train_and_save()
