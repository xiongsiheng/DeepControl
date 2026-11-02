#!/usr/bin/env python3
"""Least-busy HTTP proxy in front of several local vLLM servers.

scripts/run_local_vllm_pool.sh starts one vLLM server per GPU and this proxy in front of
them; inference clients point --base_url at the proxy and every request goes to the server
with the fewest requests in flight, so all GPUs share one pool of questions. Binds
127.0.0.1 by default (LLM_PROXY_BIND_HOST overrides it).
"""

from __future__ import annotations

import asyncio
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

BACKENDS = [v.rstrip("/") for v in os.environ["LLM_BACKENDS"].split(",") if v]
PORT = int(os.environ.get("PORT", "37300"))

app = FastAPI()
client: httpx.AsyncClient | None = None
in_flight = [0] * len(BACKENDS)
served = [0] * len(BACKENDS)
lock = asyncio.Lock()


@app.on_event("startup")
async def startup() -> None:
    global client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(900.0),
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    if client is not None:
        await client.aclose()
    print(f"[proxy] requests served per backend: {served}", flush=True)


@app.get("/proxy_stats")
async def stats() -> dict:
    return {"in_flight": in_flight, "served": served}


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request) -> Response:
    if client is None:
        return Response("proxy is not ready", status_code=503)
    async with lock:
        index = min(range(len(BACKENDS)), key=lambda i: (in_flight[i], served[i]))
        in_flight[index] += 1
        served[index] += 1
    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() in {"content-type", "accept", "authorization"}}
        upstream = await client.request(
            request.method, f"{BACKENDS[index]}/{path}", content=body, headers=headers,
            params=request.query_params,
        )
        return Response(content=upstream.content, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type"))
    finally:
        async with lock:
            in_flight[index] -= 1


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("LLM_PROXY_BIND_HOST", "127.0.0.1"), port=PORT,
                workers=1, log_level="warning")
