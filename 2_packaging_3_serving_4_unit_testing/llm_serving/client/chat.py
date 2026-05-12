"""Simple chat client for the vLLM OpenAI-compatible API."""

import logging
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("client")


def setup_hyperdx() -> None:
    try:
        from opentelemetry.sdk._logs import LoggingHandler

        from hyperdx.opentelemetry import configure_opentelemetry

        configure_opentelemetry()

        for handler in logging.getLogger().handlers:
            if isinstance(handler, LoggingHandler):
                logger.addHandler(handler)
                break

        logger.info("HyperDX telemetry configured (logs)")
    except ImportError:
        logger.warning("hyperdx-opentelemetry not installed: remote logging disabled")
    except Exception:
        logger.warning("Failed to configure HyperDX", exc_info=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")


def chat(prompt: str, model: str | None = None) -> str:
    if model is None:
        models = client.models.list()
        model = models.data[0].id

    logger.info("prompt sent: model=%s tokens=%d", model, len(prompt.split()))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else 0

    logger.info(
        "response received: %.0fms completion_tokens=%d total_tokens=%d",
        elapsed_ms,
        completion_tokens,
        total_tokens,
    )

    if elapsed_ms > 5000:
        logger.warning("slow response: %.0fms", elapsed_ms)

    return response.choices[0].message.content or ""


if __name__ == "__main__":
    setup_hyperdx()

    prompts = [
        # "Explain gradient descent in 2 sentences.",
        # "What is the capital of France?",
        "Write a Python function that reverses a string.",
    ]

    for prompt in prompts:
        print(f"\n>>> {prompt}")
        print(chat(prompt))
