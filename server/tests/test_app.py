"""Correctness tests for server/app.py's HTTP layer: request/response shape,
SSE framing, and disconnect-triggers-abort -- against FakeEngine
(server/tests/fake_engine.py), same no-CUDA-needed split as
test_engine_worker.py.

Uses FastAPI's TestClient (httpx underneath), which drives the ASGI app
(including its lifespan -- see create_app's `with TestClient(app) as
client:` below, which is what actually starts/stops the EngineWorker
thread) without a real network socket.
"""
import json
import time

from fastapi.testclient import TestClient

from server.app import create_app
from server.engine_worker import EngineWorker
from server.tests.fake_engine import FakeEngine


def make_client():
    worker = EngineWorker(FakeEngine(), idle_poll_s=0.005, default_timeout_s=5.0)
    app = create_app(worker, model_name="lightserve-test")
    return TestClient(app), worker


def parse_sse(text: str) -> list:
    """SSE frames are `data: <json>\\n\\n`; returns the decoded payloads,
    dropping the final `[DONE]` sentinel (kept as the raw string, not
    json-parsed, since it isn't JSON).
    """
    chunks = []
    for line in text.split("\n\n"):
        line = line.strip()
        if not line:
            continue
        assert line.startswith("data: ")
        payload = line[len("data: "):]
        chunks.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return chunks


class TestHealth:
    def test_health(self):
        client, _ = make_client()
        with client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestNonStreaming:
    def test_basic_completion(self):
        client, _ = make_client()
        with client:
            resp = client.post("/v1/completions", json={
                "prompt": [1, 2, 3], "max_tokens": 4,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["model"] == "lightserve-test"
        [choice] = body["choices"]
        assert choice["token_ids"] == [0, 1, 2, 3]
        assert choice["text"] == "0 1 2 3"
        assert choice["finish_reason"] == "length"

    def test_eos_maps_to_stop(self):
        client, _ = make_client()
        with client:
            resp = client.post("/v1/completions", json={
                "prompt": [1], "max_tokens": 10, "eos_token_id": 2,
            })
        [choice] = resp.json()["choices"]
        assert choice["token_ids"] == [0, 1, 2]
        assert choice["finish_reason"] == "stop"

    def test_explicit_request_id_echoed(self):
        client, _ = make_client()
        with client:
            resp = client.post("/v1/completions", json={
                "prompt": [1], "max_tokens": 1, "request_id": "my-request",
            })
        assert resp.json()["id"] == "my-request"

    def test_prompt_must_be_token_ids_not_a_string(self):
        client, _ = make_client()
        with client:
            resp = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 1})
        assert resp.status_code == 422  # pydantic rejects str where list[int] is required


class TestStreaming:
    def test_stream_yields_one_chunk_per_token_then_done(self):
        client, _ = make_client()
        with client:
            resp = client.post("/v1/completions", json={
                "prompt": [1], "max_tokens": 3, "stream": True,
            })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        chunks = parse_sse(resp.text)
        assert chunks[-1] == "[DONE]"
        token_chunks = chunks[:-1]

        token_ids = [c["choices"][0]["token_ids"][0] for c in token_chunks if c["choices"][0]["token_ids"]]
        assert token_ids == [0, 1, 2]

        finish_chunk = token_chunks[-1]
        assert finish_chunk["choices"][0]["finish_reason"] == "length"
        assert finish_chunk["choices"][0]["token_ids"] == []


class TestDisconnect:
    def test_client_disconnect_aborts_the_request(self):
        client, worker = make_client()
        with client:
            with client.stream("POST", "/v1/completions", json={
                "prompt": [1], "max_tokens": 10_000, "stream": True,
            }) as resp:
                # Read one SSE line -- proves generation actually started --
                # then let the `with` block exit without reading the rest,
                # dropping the connection early the same way a real client
                # giving up mid-stream would.
                next(resp.iter_lines())

            # Give the worker thread a moment to notice the disconnect and abort.
            for _ in range(200):
                if not worker.engine.scheduler.requests:
                    break
                time.sleep(0.01)
            assert not worker.engine.scheduler.requests, (
                "request was still tracked by the scheduler after the client disconnected"
            )
