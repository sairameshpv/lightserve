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
import textwrap  # DELIBERATE: unused import for the CI check
import json
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import Response, StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import StatusCode
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


# One JSON line per finished request (server/__main__.py prints it bare, for Loki).
request_log = logging.getLogger("lightserve.requests")


def _finish_reason(status_name: str) -> str:
    return _FINISH_REASON.get(status_name, status_name)


def _render_text(token_ids: list) -> str:
    """Not decoded text -- see schemas.py's module docstring. Space-joined
    decimal token ids, just so curl/human output isn't opaque.
    """
    return " ".join(str(t) for t in token_ids)


def create_app(worker: EngineWorker, model_name: str = "lightserve", tracer_provider=None,
               trace_sample_rate: float = 0.1, trace_slow_s: float = 1.0) -> FastAPI:
    """tracer_provider: OpenTelemetry's (default: the global one, a no-op unless configured).
    A request's trace is kept if it failed, took >= trace_slow_s, or with trace_sample_rate."""
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

    tracer = (tracer_provider or trace.get_tracer_provider()).get_tracer("lightserve")

    def trace_request(handle, start, first, last, end, n_tokens, status):
        """Spans written after the fact, from times noted while streaming -- so the keep/drop
        choice can already see whether the request was slow or failed (in-process tail sampling)."""
        failed = status == "FINISHED_ABORTED"
        if not (failed or (end - start) / 1e9 >= trace_slow_s or random.random() < trace_sample_rate):
            return None
        root = tracer.start_span("POST /v1/completions", start_time=start, attributes={
            "lightserve.request_id": handle.request_id, "lightserve.status": status, "lightserve.tokens": n_tokens})
        if failed:
            root.set_status(StatusCode.ERROR)
        context, prev = trace.set_span_in_context(root), start
        for name, at in (("queue", handle.scheduled_ns), ("prefill", first), ("decode", last), ("respond", end)):
            if at is not None:  # a request aborted early skips the phases it never reached
                span = tracer.start_span(name, context=context, start_time=prev)
                if name == "decode":
                    span.set_attribute("lightserve.tokens", n_tokens - 1)
                span.end(end_time=at)
                prev = at
        root.end(end_time=end)
        return format(root.get_span_context().trace_id, "032x")  # as Tempo displays it

    async def timed(handle, start: int):
        """handle.stream(), passed through unchanged, recording the metrics above and the
        trace. `start`: time.time_ns() at submit. All end-of-request bookkeeping happens on
        StreamEnd itself, since a streaming consumer stops iterating right after it."""
        first = last = None  # time.time_ns() of the first / latest token
        n_tokens, ended = 0, False
        in_flight.inc()
        try:
            async for item in handle.stream():
                now = time.time_ns()
                if isinstance(item, StreamToken):
                    if first is None:
                        ttft.observe((now - start) / 1e9)
                        first = now
                    else:
                        itl.observe((now - last) / 1e9)
                    tokens_total.inc()
                    last, n_tokens = now, n_tokens + 1
                else:  # StreamEnd
                    e2e.observe((now - start) / 1e9)
                    in_flight.dec()
                    ended = True
                    trace_id = trace_request(handle, start, first, last, now, n_tokens, item.finish_reason)
                    request_log.info(json.dumps({
                        "request_id": handle.request_id, "status": item.finish_reason, "tokens": n_tokens,
                        "ttft_s": None if first is None else round((first - start) / 1e9, 6),
                        "latency_s": round((now - start) / 1e9, 6), "trace_id": trace_id}))
                yield item
        finally:
            if not ended:  # abandoned mid-stream (client disconnect)
                in_flight.dec()

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest, raw_request: FastAPIRequest):
        request_id = body.request_id or f"cmpl-{uuid.uuid4().hex[:24]}"
        sampling_params = SamplingParams(max_tokens=body.max_tokens, eos_token_id=body.eos_token_id)
        start = time.time_ns()
        handle = worker.submit(
            body.prompt, sampling_params, request_id=request_id, timeout_s=body.timeout_s,
        )

        if body.stream:
            return StreamingResponse(
                _sse_stream(handle, timed(handle, start), raw_request, worker, model_name),
                media_type="text/event-stream",
            )

        token_ids: list = []
        finish_reason = None
        async for item in timed(handle, start):
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
