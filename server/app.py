"""FastAPI app exposing an OpenAI-compatible POST /v1/completions -- both
streaming (SSE) and non-streaming -- backed by an EngineWorker.

No LLMEngine import here, on purpose: `create_app()` takes an already-
constructed EngineWorker (itself wrapping *some* engine-shaped object, see
engine_worker.py's module docstring) rather than building one itself, so
this module and its tests (server/tests/test_app.py) import and run without
CUDA/Triton, same as engine_worker.py. The real LLMEngine is wired up only
in server/__main__.py, this package's one CUDA-only file.
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from engine.request import SamplingParams
from server.engine_worker import EngineWorker, StreamEnd, StreamToken
from server.schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamChunk,
)

# engine/request.py's RequestStatus member names -> OpenAI's finish_reason
# vocabulary ("stop" / "length" / a content filter reason OpenAI has that
# this engine has no equivalent of). FINISHED_ABORTED has no OpenAI
# equivalent (client disconnect or this server's own timeout, not a model
# decision) -- surfaced as "abort" rather than silently mapped to one of
# theirs.
_FINISH_REASON = {
    "FINISHED_STOPPED": "stop",
    "FINISHED_LENGTH_CAPPED": "length",
    "FINISHED_ABORTED": "abort",
}


def _finish_reason(status_name: str) -> str:
    return _FINISH_REASON.get(status_name, status_name)


def _render_text(token_ids: list) -> str:
    """Not decoded text -- see schemas.py's module docstring. Space-joined
    decimal token ids, just so curl/human output isn't opaque.
    """
    return " ".join(str(t) for t in token_ids)


def create_app(worker: EngineWorker, model_name: str = "lightserve") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="lightserve", lifespan=lifespan)

    # The worker's registry (one per worker, not prometheus_client's global
    # one, so the many workers tests create don't collide): one /metrics page
    # for engine-side and HTTP-side metrics. Requests ended are counted there,
    # since only the worker sees every ending (disconnects, timeouts included).
    registry = worker.registry
    tokens_total = Counter("lightserve_generated_tokens", "Tokens generated", registry=registry)
    ttft = Histogram("lightserve_time_to_first_token_seconds", "Submit to first token", registry=registry)
    itl = Histogram("lightserve_inter_token_seconds", "Gap between consecutive tokens", registry=registry)
    e2e = Histogram("lightserve_request_seconds", "Submit to finish", registry=registry)
    in_flight = Gauge("lightserve_requests_in_flight", "Requests being streamed now", registry=registry)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    async def timed(handle):
        """handle.stream(), passed through unchanged, recording the metrics above."""
        start, last = time.perf_counter(), None  # last: previous token's time, None before the first
        in_flight.inc()
        try:
            async for item in handle.stream():
                now = time.perf_counter()
                if isinstance(item, StreamToken):
                    if last is None:
                        ttft.observe(now - start)
                    else:
                        itl.observe(now - last)
                    tokens_total.inc()
                    last = now
                else:  # StreamEnd
                    e2e.observe(now - start)
                yield item
        finally:
            in_flight.dec()

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest, raw_request: FastAPIRequest):
        request_id = body.request_id or f"cmpl-{uuid.uuid4().hex[:24]}"
        sampling_params = SamplingParams(max_tokens=body.max_tokens, eos_token_id=body.eos_token_id)
        handle = worker.submit(
            body.prompt, sampling_params, request_id=request_id, timeout_s=body.timeout_s,
        )

        if body.stream:
            return StreamingResponse(
                _sse_stream(handle, timed(handle), raw_request, worker, model_name),
                media_type="text/event-stream",
            )

        token_ids: list = []
        finish_reason = None
        async for item in timed(handle):
            if isinstance(item, StreamToken):
                token_ids.append(item.token_id)
            elif isinstance(item, StreamEnd):
                finish_reason = _finish_reason(item.finish_reason)
        return CompletionResponse(
            id=request_id,
            model=model_name,
            choices=[CompletionChoice(
                index=0, text=_render_text(token_ids), token_ids=token_ids, finish_reason=finish_reason,
            )],
        )

    return app


async def _sse_stream(handle, stream, raw_request: FastAPIRequest, worker: EngineWorker, model_name: str):
    """One SSE frame per token, a final frame carrying finish_reason, then
    the OpenAI-convention `data: [DONE]` sentinel.

    Polls both the stream and `raw_request.is_disconnected()` on the same
    cadence (wrapping the stream's own `__anext__()` in `asyncio.wait_for`)
    rather than only checking between yields -- a naive "check once per
    token" would leave a disconnect undetected for however long the *next*
    token takes to generate, which is exactly the case (a client that gave
    up, still holding a decode slot/KV-cache blocks) this check exists to
    catch quickly. On disconnect, aborts the request so its resources free
    immediately instead of running to completion (or its timeout) with
    nobody listening.
    """
    stream_iter = stream.__aiter__()
    try:
        while True:
            if await raw_request.is_disconnected():
                worker.abort(handle.request_id)
                return
            try:
                item = await asyncio.wait_for(stream_iter.__anext__(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                break

            if isinstance(item, StreamToken):
                chunk = CompletionStreamChunk(
                    id=handle.request_id, model=model_name,
                    choices=[CompletionStreamChoice(
                        index=0, text=_render_text([item.token_id]), token_ids=[item.token_id],
                    )],
                )
            else:  # StreamEnd
                chunk = CompletionStreamChunk(
                    id=handle.request_id, model=model_name,
                    choices=[CompletionStreamChoice(
                        index=0, text="", token_ids=[], finish_reason=_finish_reason(item.finish_reason),
                    )],
                )
            yield f"data: {chunk.model_dump_json()}\n\n"
            if isinstance(item, StreamEnd):
                break
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        # ASGI server tore down the connection out from under us (distinct
        # from raw_request.is_disconnected(), which only reflects what the
        # client has told us so far) -- same cleanup as a detected
        # disconnect, then re-raise so the server finishes its own teardown.
        worker.abort(handle.request_id)
        raise
