from fastapi import FastAPI

app = FastAPI(title="Simple API", version="0.1.0")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Hello, CHANGED AGAIN World"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Single process: Ctrl+C sends SIGINT to uvicorn and exits. With reload=True,
    # uvicorn runs a parent watcher + child worker; signals often misbehave in
    # `docker exec` shells. For auto-reload on your laptop: uvicorn app:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
