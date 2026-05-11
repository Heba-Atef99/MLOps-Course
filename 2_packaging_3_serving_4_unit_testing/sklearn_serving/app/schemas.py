from dataclasses import dataclass


@dataclass
class PredictRequest:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@dataclass
class PredictResponse:
    class_name: str
    class_index: int
    probabilities: dict[str, float]


@dataclass
class HealthResponse:
    status: str
    model_loaded: bool
