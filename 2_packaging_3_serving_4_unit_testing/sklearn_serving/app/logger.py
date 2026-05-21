import logging
from litestar.logging import LoggingConfig

logger = logging.getLogger("app")

def setup_hyperdx() -> None:
    try:
        from hyperdx.opentelemetry import configure_opentelemetry

        configure_opentelemetry()

        logger.info("HyperDX telemetry configured (logs + traces)")
    except ImportError:
        logger.warning("hyperdx-opentelemetry not installed: remote logging disabled")
    except Exception:
        logger.warning("Failed to configure HyperDX", exc_info=True)


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
        "app": {"level": "DEBUG", "handlers": [], "propagate": True},
    },
)
