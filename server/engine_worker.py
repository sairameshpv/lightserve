"""Background engine loop plus the request-queuing/timeout/streaming
plumbing that sits between FastAPI's async request handlers and
LLMEngine.step(). See server/README.md for the full producer/consumer
picture; this module is the consumer-side half (app.py is the producer
side, HTTP in).

Deliberately torch-free, same discipline as engine/'s own modules (see
engine/request.py's module docstring): everything here touches only an
LLMEngine-*shaped* interface --

    engine.add_request(prompt_token_ids, sampling_params, request_id) -> Request
    engine.step() -> (SchedulerOutput, finished: list[Request])
    engine.scheduler.has_unfinished_requests() -> bool
    engine.scheduler.abort_requests(request_ids) -> list[Request]

-- never model/kv_cache.py, model/model_runner.py, or a real GPU directly.
That means this file and its tests (server/tests/test_engine_worker.py) run
without CUDA/Triton at all, against a fake engine built from the real,
already-torch-free engine/scheduler.py -- the same "verify the CPU-side
logic for real, the GPU-side statically" split model/llm_engine.py's own
tests already accept for the pieces that do need a GPU. The real LLMEngine
is only ever imported in server/__main__.py.

Threading model, and why: LLMEngine.step() is a synchronous, GPU-bound call.
It must never run on the asyncio event loop FastAPI's handlers share --
doing so would stall every other in-flight request's I/O (including reading
tokens back out of *other* requests' streams) for the duration of each
forward pass. So the engine lives on exactly one dedicated background
thread that does nothing else, looping:

    drain queued submissions/aborts -> sweep timed-out requests -> step()
        -> fan this step's new tokens out to each request's own queue

FastAPI's handlers never call into the engine directly; they only push onto
(submit/abort) or read from (StreamHandle) plain thread-safe queues this
class owns. `engine`'s own state (Scheduler's waiting/running lists,
BlockManager's free list) is therefore only ever touched from this one
thread, so none of it needs its own locking.
"""
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamToken:
    """One newly-generated token for a request, in generation order."""
    token_id: int


@dataclass
class StreamEnd:
    """Terminal marker for a request's stream. `finish_reason` is the
    engine-side RequestStatus member name (e.g. "FINISHED_STOPPED",
    "FINISHED_LENGTH_CAPPED", "FINISHED_ABORTED") -- both an explicit
    `EngineWorker.abort()` call and a timeout sweep go through
    Scheduler.abort_requests, so engine/request.py's RequestStatus has no
    separate "timed out" state to report here; a caller that cares why an
    abort happened has to track that itself (server/app.py doesn't need
    to).
    """
    finish_reason: str


class StreamHandle:
    """One request's channel out of the engine thread. Backed by a plain
    queue.SimpleQueue, not an asyncio.Queue -- the engine thread that
    produces into it has no event loop of its own to call asyncio-queue-safe
    methods from. `stream()` is the async consumer side (used by
    server/app.py's request handlers): it polls rather than blocking,
    because SimpleQueue has no async-native wait, at an interval short
    enough to not be a user-visible add to per-token streaming latency (well
    under one model forward pass).
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.queue: "queue.SimpleQueue" = queue.SimpleQueue()

    async def stream(self, poll_interval: float = 0.01):
        import asyncio
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(poll_interval)
                continue
            yield item
            if isinstance(item, StreamEnd):
                return


@dataclass
class _Submission:
    prompt_token_ids: list
    sampling_params: object
    request_id: str
    timeout_s: Optional[float]
    handle: StreamHandle


@dataclass
class _Abort:
    request_id: str


class EngineWorker:
    """Owns one LLMEngine-shaped `engine` and the one background thread
    allowed to touch it. See module docstring for the full design.

    default_timeout_s: applied to a submission that doesn't specify its own
    `timeout_s` -- the backstop against a request that's stuck (never
    scheduled because the cache is permanently full, or generating far
    longer than expected) holding a decode slot / KV-cache blocks forever
    with no client left listening. `timeout_s=None` on a submission (not the
    default) opts a request out of the backstop entirely.
    """

    def __init__(self, engine, idle_poll_s: float = 0.005, default_timeout_s: Optional[float] = 60.0):
        self.engine = engine
        self.idle_poll_s = idle_poll_s
        self.default_timeout_s = default_timeout_s
        self._incoming: "queue.SimpleQueue" = queue.SimpleQueue()
        self._streams: dict = {}    # request_id -> StreamHandle, engine-thread-only after _handle_submission
        self._deadlines: dict = {}  # request_id -> absolute time.time() deadline, engine-thread-only
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)

    # -- Producer side: called from any thread (FastAPI's event loop, in practice) ----

    def submit(self, prompt_token_ids: list, sampling_params, request_id: Optional[str] = None,
               timeout_s: Optional[float] = None) -> StreamHandle:
        """Returns a StreamHandle immediately. The actual
        `engine.add_request()` call still only happens later, on the engine
        thread once it drains this submission -- so two concurrent submit()
        calls from different request handlers can never race on the
        engine's own (single-threaded-assumed) state. `request_id` is
        resolved here, synchronously, rather than left to
        `engine.add_request`'s own default-id logic, precisely so the
        caller can know it immediately (needed for e.g. `abort()` on client
        disconnect) without waiting on the engine thread.
        """
        request_id = request_id or uuid.uuid4().hex
        handle = StreamHandle(request_id)
        self._incoming.put(_Submission(
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            timeout_s=timeout_s if timeout_s is not None else self.default_timeout_s,
            handle=handle,
        ))
        return handle

    def abort(self, request_id: str) -> None:
        """Called from any thread -- e.g. server/app.py noticing its client
        disconnected mid-stream. Just enqueues; the actual
        `scheduler.abort_requests` call happens on the engine thread, for
        the same reason submit() defers add_request.
        """
        self._incoming.put(_Abort(request_id))

    # -- Engine thread ------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain_incoming()
            self._sweep_timeouts()
            if self.engine.scheduler.has_unfinished_requests():
                output, finished = self.engine.step()
                self._emit_tokens(output)
                self._finish(finished)
            else:
                time.sleep(self.idle_poll_s)

    def _drain_incoming(self) -> None:
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _Submission):
                self._handle_submission(item)
            else:  # _Abort
                self._finish(self.engine.scheduler.abort_requests([item.request_id]))

    def _handle_submission(self, submission: _Submission) -> None:
        request = self.engine.add_request(
            submission.prompt_token_ids, submission.sampling_params, submission.request_id,
        )
        request.arrival_time = time.time()  # engine.add_request doesn't set this -- Request's own default is 0.0
        self._streams[submission.request_id] = submission.handle
        if submission.timeout_s is not None:
            self._deadlines[submission.request_id] = request.arrival_time + submission.timeout_s

    def _sweep_timeouts(self) -> None:
        now = time.time()
        expired = [rid for rid, deadline in self._deadlines.items() if now >= deadline]
        if expired:
            self._finish(self.engine.scheduler.abort_requests(expired))

    def _emit_tokens(self, scheduler_output) -> None:
        for sr in list(scheduler_output.scheduled_new) + list(scheduler_output.scheduled_running):
            request = sr.request
            handle = self._streams.get(request.request_id)
            if handle is not None and request.output_token_ids:
                handle.queue.put(StreamToken(request.output_token_ids[-1]))

    def _finish(self, requests: list) -> None:
        """Shared teardown for every way a request stops being tracked --
        a natural finish (from step()'s own free_finished_requests sweep),
        an explicit abort(), or a timeout. `abort_requests` already frees
        the request's blocks and removes it from the scheduler entirely
        (see engine/scheduler.py); this just tears down the worker's own
        bookkeeping and unblocks whatever's reading the StreamHandle.
        """
        for request in requests:
            self._deadlines.pop(request.request_id, None)
            handle = self._streams.pop(request.request_id, None)
            if handle is not None:
                handle.queue.put(StreamEnd(finish_reason=request.status.name))
