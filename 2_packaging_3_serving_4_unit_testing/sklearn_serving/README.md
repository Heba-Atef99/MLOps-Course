# Iris Serving Demo: sklearn + Litestar

A minimal MLOps demo: train an Iris classifier, package it safely, serve it via API, log everything, and test it.

## Quick Start

```bash
uv sync                          # install deps
uv run python training/train.py  # train & serialize model
uv run uvicorn app.main:app --reload  # start server
uv run pytest                    # run tests
```

**API docs (while server is running):**

| URL | Description |
| --- | --- |
| http://localhost:8000/schema/swagger | Swagger UI |
| http://localhost:8000/schema/redoc | ReDoc |
| http://localhost:8000/schema/rapidoc | RapiDoc |
| http://localhost:8000/schema/elements | Stoplight Elements |
| http://localhost:8000/schema/openapi.json | Raw OpenAPI JSON |

## Project Structure

```
sklearn_serving/
├── training/          # OFFLINE: never deployed
│   └── train.py       # Train RandomForest, serialize with skops
├── app/               # DEPLOYED: serving only
│   ├── main.py        # Litestar app (3 endpoints + logging)
│   ├── model.py       # Safe model loading & prediction
│   └── schemas.py     # Dataclass request/response schemas
├── tests/
│   ├── test_model.py  # 3 function tests
│   └── test_api.py    # 4 endpoint tests
└── model/             # Serialized model artifact (1.8MB)
```

## Endpoints

| Method | Path       | Description             |
| ------ | ---------- | ----------------------- |
| GET    | `/`        | Welcome + endpoint list |
| GET    | `/health`  | Health check            |
| POST   | `/predict` | Iris class prediction   |

## Lessons Learned & Concepts Applied

### Packaging

- **skops** for safe sklearn serialization. Unlike pickle/joblib, it refuses unknown types by default and cannot execute arbitrary code on load
- **Train/deploy separation**: training code (`training/`) never ships with the server (`app/`). Different concerns, different lifecycles
- **uv** for dependency management with `uv.lock` committed for reproducible installs across machines

### Serving

- **Litestar** over FastAPI: native dataclass support (no Pydantic required), msgspec for fast serialization, correct HTTP semantics (POST returns 201), 4 OpenAPI UIs out of the box
- **Pydantic not needed**: plain Python dataclasses with type annotations give you automatic request validation via Litestar's msgspec layer

### Logging

- **All 5 logging levels** used deliberately:
  - `DEBUG`: input features, inference timing, model path
  - `INFO`: predictions, startup, health checks
  - `WARNING`: slow inference (>100ms), missing HyperDX
  - `ERROR`: prediction when model not loaded
  - `CRITICAL`: model file missing at startup
- **Dual output**: stdout (local dev) + HyperDX via OpenTelemetry (live monitoring)

### Unit Testing

- **Function tests**: prediction logic tested independently from the API
- **Endpoint tests**: Litestar's built-in `TestClient` for HTTP-level testing
- **71% coverage** (100% on model + schemas, 78% on main app)

### Code Quality

- **Pre-commit** with ruff (replaces black + isort: they conflict) and pyright
- **ruff** handles linting, import sorting, and formatting in one tool
