import time
from pathlib import Path

from dotenv import load_dotenv
from litestar import Litestar, get, post
from litestar.datastructures import State
from litestar.exceptions import HTTPException

from app.logger import logger, logging_config, setup_hyperdx
from app.model import load_model, predict
from app.schemas import HealthResponse, PredictRequest, PredictResponse

load_dotenv()

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "loan_model.skops"


def on_startup(app: Litestar) -> None:
    setup_hyperdx()

    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
        from opentelemetry.trace import get_tracer_provider

        app.asgi_handler = OpenTelemetryMiddleware(  # type: ignore[assignment]
            app.asgi_handler,
            tracer_provider=get_tracer_provider(),
        )
        logger.info("ASGI tracing middleware enabled")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-asgi not installed: tracing disabled")

    logger.info("Starting Loan Approval Serving Demo")
    logger.debug("Model path: %s", MODEL_PATH)

    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.critical(
            "Model not found at %s: run 'uv run python training/train.py' first",
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
        "message": "Loan Approval Prediction API",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /predict": "Predict loan approval",
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
    logger.info(
        "Prediction requested: cibil=%d income=%d loan=%d",
        data.cibil_score,
        data.income_annum,
        data.loan_amount,
    )

    model = getattr(state, "model", None)
    if model is None:
        logger.error("Prediction failed: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    features = {
        "no_of_dependents": data.no_of_dependents,
        "education": data.education,
        "self_employed": data.self_employed,
        "income_annum": data.income_annum,
        "loan_amount": data.loan_amount,
        "loan_term": data.loan_term,
        "cibil_score": data.cibil_score,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value,
    }
    result = predict(model, features)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if elapsed_ms > 100:
        logger.warning("Slow inference detected: %.2fms", elapsed_ms)
    else:
        logger.debug("Inference completed in %.2fms", elapsed_ms)

    logger.info(
        "Prediction: approved=%s probability=%.2f%%",
        result["loan_approved"],
        result["approval_probability"] * 100,
    )

    return PredictResponse(
        loan_approved=result["loan_approved"],
        approval_probability=result["approval_probability"],
    )


app = Litestar(
    route_handlers=[home, health_check, predict_endpoint],
    on_startup=[on_startup],
    logging_config=logging_config,
)
