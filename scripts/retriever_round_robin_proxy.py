#!/usr/bin/env python3
"""Small round-robin proxy for independent CPU retriever backends."""

from __future__ import annotations

import asyncio
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response


BACKENDS = [
    value.rstrip("/")
    for value in os.environ.get(
        "RETRIEVER_BACKENDS",
        "http://127.0.0.1:8201,http://127.0.0.1:8202,http://127.0.0.1:8203",
    ).split(",")
    if value
]
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI()
client: httpx.AsyncClient | None = None
backend_index = 0
backend_lock = asyncio.Lock()


@app.on_event("startup")
async def startup() -> None:
    global client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(180.0),
        limits=httpx.Limits(
            max_connections=512,
            max_keepalive_connections=256,
        ),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    if client is not None:
        await client.aclose()


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(path: str, request: Request) -> Response:
    global backend_index
    if client is None:
        return Response("proxy is not ready", status_code=503)

    async with backend_lock:
        start = backend_index
        backend_index = (backend_index + 1) % len(BACKENDS)

    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in {"content-type", "accept"}
    }
    last_error: Exception | None = None
    for offset in range(len(BACKENDS)):
        backend = BACKENDS[(start + offset) % len(BACKENDS)]
        try:
            upstream = await client.request(
                request.method,
                f"{backend}/{path}",
                content=body,
                headers=headers,
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
            )
        except httpx.HTTPError as error:
            last_error = error

    return Response(
        f"all retriever backends failed: {last_error}",
        status_code=503,
    )


if __name__ == "__main__":
    # Loopback by default; RETRIEVER_BIND_HOST overrides it for a cross-node setup.
    uvicorn.run(app, host=os.environ.get("RETRIEVER_BIND_HOST", "127.0.0.1"), port=PORT, workers=1)
