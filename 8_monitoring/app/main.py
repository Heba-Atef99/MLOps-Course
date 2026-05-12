import logging
import time
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litestar import Litestar, Request, get, post
from litestar.datastructures import State
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig
from litestar.middleware.base import AbstractMiddleware

from app.axiom_client import send_events
from app.model import load_model, predict
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

load_dotenv()

logger = logging.getLogger("app")

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "ctr_model.skops"

MAX_PENDING = 10000
pending_predictions: OrderedDict[str, dict] = OrderedDict()


class AxiomMiddleware(AbstractMiddleware):
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        start = time.perf_counter()

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                duration = time.perf_counter() - start
                send_events(
                    [
                        {
                            "_time": datetime.now(UTC).isoformat(),
                            "event_type": "http_request",
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": message["status"],
                            "duration_seconds": round(duration, 6),
                        }
                    ]
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


def setup_hyperdx() -> None:
    try:
        from hyperdx.opentelemetry import configure_opentelemetry

        configure_opentelemetry()
        logger.info("HyperDX telemetry configured")
    except ImportError:
        logger.warning("hyperdx-opentelemetry not installed: remote logging disabled")
    except Exception:
        logger.warning("Failed to configure HyperDX", exc_info=True)


def on_startup(app: Litestar) -> None:
    setup_hyperdx()

    logger.info("Starting CTR Monitoring Demo")
    logger.debug("Model path: %s", MODEL_PATH)

    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.critical(
            "Model not found at %s: run 'uv run python training/train.py'",
            MODEL_PATH,
        )
        raise
    except Exception:
        logger.critical("Failed to load model", exc_info=True)
        raise


@get("/")
async def home() -> dict[str, object]:
    logger.debug("Home endpoint accessed")
    return {
        "message": "CTR Monitoring Demo",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /predict": "Predict click-through",
            "POST /feedback": "Submit actual click result",
        },
    }


@get("/health")
async def health_check(state: State) -> HealthResponse:
    model_loaded = getattr(state, "model", None) is not None
    status = "healthy" if model_loaded else "unhealthy"

    if not model_loaded:
        logger.warning("Health check failed: model not loaded")
    else:
        logger.info("Health check: %s", status)

    return HealthResponse(status=status, model_loaded=model_loaded)


@post("/predict")
async def predict_endpoint(data: PredictRequest, state: State) -> PredictResponse:
    model = getattr(state, "model", None)
    if model is None:
        logger.error("Prediction failed: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    features = [
        data.hour_of_day,
        data.device_type,
        data.ad_position,
        data.user_age,
        data.session_duration_sec,
        data.page_views,
    ]
    result = predict(model, features)
    elapsed_ms = (time.perf_counter() - start) * 1000

    prediction_id = str(uuid.uuid4())

    pending_predictions[prediction_id] = {
        "predicted_click": result["predicted_click"],
        "confidence": result["confidence"],
    }
    if len(pending_predictions) > MAX_PENDING:
        pending_predictions.popitem(last=False)

    logger.info(
        "Prediction %s: click=%s confidence=%.2f (%.1fms)",
        prediction_id[:8],
        result["predicted_click"],
        result["confidence"],
        elapsed_ms,
    )

    send_events(
        [
            {
                "_time": datetime.now(UTC).isoformat(),
                "event_type": "prediction",
                "prediction_id": prediction_id,
                "predicted_click": result["predicted_click"],
                "confidence": result["confidence"],
                "click_probability": result["click_probability"],
                "latency_seconds": round(elapsed_ms / 1000, 6),
                "feature_hour_of_day": data.hour_of_day,
                "feature_device_type": data.device_type,
                "feature_ad_position": data.ad_position,
                "feature_user_age": data.user_age,
                "feature_session_duration_sec": data.session_duration_sec,
                "feature_page_views": data.page_views,
            }
        ]
    )

    return PredictResponse(
        prediction_id=prediction_id,
        predicted_click=result["predicted_click"],
        confidence=result["confidence"],
    )


@post("/feedback")
async def feedback_endpoint(data: FeedbackRequest) -> FeedbackResponse:
    prediction = pending_predictions.pop(data.prediction_id, None)
    if prediction is None:
        logger.warning("Feedback for unknown prediction: %s", data.prediction_id[:8])
        raise HTTPException(
            status_code=404, detail="Prediction not found or already received feedback"
        )

    correct = prediction["predicted_click"] == data.clicked

    logger.info(
        "Feedback %s: predicted=%s actual=%s correct=%s",
        data.prediction_id[:8],
        prediction["predicted_click"],
        data.clicked,
        correct,
    )

    send_events(
        [
            {
                "_time": datetime.now(UTC).isoformat(),
                "event_type": "feedback",
                "prediction_id": data.prediction_id,
                "predicted_click": prediction["predicted_click"],
                "actual_click": data.clicked,
                "correct": correct,
                "confidence": prediction["confidence"],
            }
        ]
    )

    return FeedbackResponse(
        prediction_id=data.prediction_id,
        predicted_click=prediction["predicted_click"],
        actual_click=data.clicked,
        correct=correct,
    )


logging_config = LoggingConfig(
    root={"level": "INFO", "handlers": ["console"]},
    formatters={
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        },
    },
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    loggers={
        "app": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
    },
)

app = Litestar(
    route_handlers=[home, health_check, predict_endpoint, feedback_endpoint],
    on_startup=[on_startup],
    logging_config=logging_config,
    middleware=[AxiomMiddleware],
)
