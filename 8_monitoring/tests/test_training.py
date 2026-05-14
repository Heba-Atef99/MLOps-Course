import numpy as np

from training import train


def test_generate_synthetic_data_shapes_and_binary_labels():
    X, y = train.generate_synthetic_data(n_samples=20, seed=123)

    assert X.shape == (20, 6)
    assert y.shape == (20,)
    assert set(np.unique(y)).issubset({0, 1})


def test_train_and_save_writes_model_and_training_baseline(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    data_dir = tmp_path / "data"
    baseline_path = data_dir / "training_baseline.csv"

    monkeypatch.setattr(train, "MODEL_DIR", model_dir)
    monkeypatch.setattr(train, "DATA_DIR", data_dir)
    monkeypatch.setattr(train, "BASELINE_PATH", baseline_path)
    generate_synthetic_data = train.generate_synthetic_data
    monkeypatch.setattr(
        train,
        "generate_synthetic_data",
        lambda: generate_synthetic_data(n_samples=200, seed=42),
    )

    train.train_and_save()

    assert (model_dir / "ctr_model.skops").exists()
    assert baseline_path.exists()
    header = baseline_path.read_text().splitlines()[0]
    assert header == ",".join(train.FEATURE_NAMES)
