from dataclasses import dataclass


@dataclass
class PredictRequest:
    hour_of_day: int
    device_type: int
    ad_position: int
    user_age: int
    session_duration_sec: float
    page_views: int


@dataclass
class PredictResponse:
    prediction_id: str
    predicted_click: bool
    confidence: float


@dataclass
class FeedbackRequest:
    prediction_id: str
    clicked: bool


@dataclass
class FeedbackResponse:
    prediction_id: str
    predicted_click: bool
    actual_click: bool
    correct: bool


@dataclass
class HealthResponse:
    status: str
    model_loaded: bool
