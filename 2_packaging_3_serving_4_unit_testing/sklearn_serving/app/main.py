import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from litestar import Litestar, get, post
from litestar.datastructures import State
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig

from app.model import load_model, predict
from app.schemas import HealthResponse, PredictRequest, PredictResponse

load_dotenv()

logger = logging.getLogger("app")

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.skops"


def setup_hyperdx() -> None:
    try:
        from hyperdx.opentelemetry import configure_opentelemetry

        configure_opentelemetry()
        logger.info("HyperDX telemetry configured")
    except ImportError:
        logger.warning("hyperdx-opentelemetry not installed — remote logging disabled")
    except Exception:
        logger.warning("Failed to configure HyperDX", exc_info=True)


def on_startup(app: Litestar) -> None:
    setup_hyperdx()

    logger.info("Starting Iris Serving Demo")
    logger.debug("Model path: %s", MODEL_PATH)

    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.critical(
            "Model file not found at %s — run 'uv run python training/train.py' first",
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
        "message": "Welcome to the Iris Serving Demo",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /predict": "Make a prediction",
        },
    }


@get("/health")
async def health_check(state: State) -> HealthResponse:
    model_loaded = getattr(state, "model", None) is not None
    status = "healthy" if model_loaded else "unhealthy"

    if not model_loaded:
        logger.warning("Health check failed — model not loaded")
    else:
        logger.info("Health check: %s", status)

    return HealthResponse(status=status, model_loaded=model_loaded)


@post("/predict")
async def predict_endpoint(data: PredictRequest, state: State) -> PredictResponse:
    logger.info(
        "Prediction requested — features: [%.2f, %.2f, %.2f, %.2f]",
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    )

    model = getattr(state, "model", None)
    if model is None:
        logger.error("Prediction failed — model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]
    result = predict(model, features)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if elapsed_ms > 100:
        logger.warning("Slow inference detected: %.2fms", elapsed_ms)
    else:
        logger.debug("Inference completed in %.2fms", elapsed_ms)

    logger.info(
        "Prediction: %s (confidence: %.1f%%)",
        result["class_name"],
        max(result["probabilities"].values()) * 100,
    )

    return PredictResponse(
        class_name=result["class_name"],
        class_index=result["class_index"],
        probabilities=result["probabilities"],
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
    route_handlers=[home, health_check, predict_endpoint],
    on_startup=[on_startup],
    logging_config=logging_config,
)
