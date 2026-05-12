import json
import logging
import os

from axiom_py import Client as AxiomClient
from axiom_py.client import ContentEncoding, ContentType
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("app")

_client: AxiomClient | None = None


def _get_dataset() -> str:
    return os.getenv("AXIOM_DATASET", "mlops")


def get_client() -> AxiomClient | None:
    global _client
    if _client is None:
        try:
            _client = AxiomClient()
            logger.info("Axiom client initialized")
        except Exception as e:
            logger.warning("Axiom client not initialized: %s", e)
    return _client


def send_events(events: list[dict]) -> None:
    client = get_client()
    if not client:
        return
    try:
        payload = json.dumps(events).encode("utf-8")
        client.ingest(
            _get_dataset(), payload, ContentType.JSON, ContentEncoding.IDENTITY
        )
    except Exception as e:
        logger.warning("Failed to send events to Axiom: %s", e)
