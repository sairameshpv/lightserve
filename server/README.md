# server/

An OpenAI-compatible HTTP front end for `model/llm_engine.py`'s `LLMEngine`:
a FastAPI app exposing `POST /v1/completions`, streaming and non-streaming,
with request queuing and timeout handling in front of it.

## Why this can't just call `LLMEngine` directly from a request handler

`LLMEngine.step()` is synchronous and GPU-bound -- one `schedule() ->
execute_model() -> free_finished_requests()` call per iteration (see
`model/llm_engine.py`). FastAPI's request handlers, though, run on one
shared asyncio event loop: if a handler called `engine.step()` itself,
every other in-flight request -- including ones just trying to read their
*own* already-generated tokens back out -- would stall for the entire
duration of that forward pass. And `Scheduler`'s own state (`waiting`/
`running`, `BlockManager`'s free list) isn't designed to be touched from
multiple threads at once, so two concurrent handlers can't safely share it
either.

So this package makes the engine a single-threaded resource with its own
dedicated background thread (`EngineWorker` in `engine_worker.py`), and HTTP
handlers only ever talk to it through thread-safe queues:

```
        submit()                    EngineWorker._run(), one background thread
HTTP  ─────────────►  queue.SimpleQueue   ┌───────────────────────────────┐
handler   (StreamHandle              │ loop:                          │
           returned                  │   drain submissions/aborts     │
           immediately)              │   sweep timed-out requests     │
                                      │   engine.step()                │
                                      │   fan new tokens out to each   │
                                      │     request's own queue        │
                                      └───────────────────────────────┘
HTTP  ◄─────────────  StreamHandle.queue (one queue.SimpleQueue per request)
handler    (async `stream()` polls this and yields StreamToken/StreamEnd)
```

This is "request queuing" at two levels, worth keeping distinct:

1. **HTTP admission queue** (`EngineWorker._incoming`) -- submissions
   waiting for the background thread to notice them and call
   `scheduler.add_request()`. Bounded only by memory; nothing here rejects
   a submission for being queued too long except the timeout below.
2. **Continuous-batching queue** -- `Scheduler`'s own `waiting`/`running`
   lists (`engine/scheduler.py`), unchanged and untouched by this package.
   This is what interleaves multiple prompts' prefill/decode steps once
   they've been admitted; this package only ever calls `add_request()` and
   `step()`, the same public surface `LLMEngine.generate()` itself uses.

## Files

- **`engine_worker.py`** -- `EngineWorker`, the background thread plus
  submit/abort/timeout plumbing described above. Deliberately torch-free:
  it only calls an `LLMEngine`-*shaped* interface (`add_request`, `step`,
  `scheduler.has_unfinished_requests`, `scheduler.abort_requests`), never
  imports `model/` or `kernels/` directly. That's what lets
  `tests/test_engine_worker.py` run for real, without CUDA/Triton, against
  `tests/fake_engine.py`'s `FakeEngine` -- a stand-in built on the real,
  already-torch-free `engine/scheduler.py`, with a fake `step()` that
  mimics `ModelRunner.execute_model`'s contract (append a token to
  `output_token_ids`, call `maybe_finish()`) without a real forward pass.
- **`schemas.py`** -- Pydantic request/response models for
  `POST /v1/completions`. No tokenizer in this repo (see
  `model/llm_engine.py`'s `generate()` docstring), so `prompt` uses OpenAI's
  own array-of-token-ids mode rather than a string -- see this file's module
  docstring for why that's the one OpenAI-compatible option that doesn't
  need a tokenizer. `text` in every response is consequently *not* decoded
  text, just a space-joined string of the token ids for legibility.
- **`app.py`** -- `create_app(worker)`: the FastAPI routes themselves
  (`/health`, `/v1/completions`), SSE framing for the streaming path, and
  disconnect handling (aborts the request the moment a streaming client
  goes away, rather than letting it run to completion or its timeout with
  nobody listening). Also torch-free, for the same reason and the same
  payoff -- `tests/test_app.py` drives the real ASGI app end to end
  (`FastAPI TestClient`, no real socket) against `FakeEngine`.
- **`__main__.py`** -- `python -m server`: the one CUDA-only file in this
  package. Constructs a real `LLMEngine` (via `model/minimal_llama.py`'s
  `llama3_8b_shape()`) and serves it with uvicorn. Not imported by
  `engine_worker.py` or `app.py` or their tests -- only this file pulls in
  `model/kv_cache.py` -> `model/minimal_llama.py` ->
  `kernels/flash_attention.py`'s `import triton`, so it being
  CUDA/Triton-dependent at import time doesn't block running the rest of
  this package's tests on a machine without a GPU (this repo's own `triton`
  has no macOS wheel -- see `model/README.md`). Written and statically
  reviewed here; not yet run for real -- next step is the L40S, same as
  every other CUDA-only piece in this repo.

## Request lifecycle, end to end

1. `POST /v1/completions` arrives. `app.py` resolves a `request_id`
   (client-supplied or generated) and calls `worker.submit(...)`, which
   returns a `StreamHandle` immediately -- the actual `engine.add_request()`
   call happens later, on the engine thread, once it drains the submission.
2. Non-streaming: the handler `async for`s the `StreamHandle` until
   `StreamEnd`, buffering every `StreamToken`, then returns one
   `CompletionResponse`.
3. Streaming: the handler wraps the same iteration in a
   `StreamingResponse`, emitting one `data: {...}\n\n` SSE frame per token,
   a final frame carrying `finish_reason`, then `data: [DONE]\n\n` -- the
   OpenAI streaming convention. `curl -N` (no buffering) against this shows
   tokens arriving one at a time rather than curl waiting for the whole
   body.
4. Meanwhile, on the background thread: `EngineWorker._run()` loops
   `drain submissions/aborts -> sweep timeouts -> engine.step() -> fan out
   tokens`, forever, sleeping briefly whenever `scheduler.has_unfinished_
   requests()` is `False` rather than busy-spinning.
5. A request stops being tracked exactly one of three ways, all funneled
   through `EngineWorker._finish()`: it naturally finishes (max_tokens or
   eos, surfaced by `step()`'s own `free_finished_requests()` sweep), it's
   explicitly aborted (`worker.abort(request_id)` -- what a streaming
   client's disconnect triggers), or it times out (see below). All three
   go through `Scheduler.abort_requests`/`free_finished_requests`, so
   `engine/request.py`'s `RequestStatus` has no separate "timed out" state
   -- a timeout and an explicit abort both surface as
   `FINISHED_ABORTED`, mapped to OpenAI's `finish_reason` vocabulary as
   `"abort"` (an addition to OpenAI's own `"stop"`/`"length"`, since neither
   fits "the client gave up" or "this server's own backstop fired").

## Timeout handling

Two different timeouts, easy to conflate, handled at two different layers:

- **Per-request backstop** (`EngineWorker`'s `default_timeout_s`, or a
  request's own `timeout_s`) -- guards against a request stuck holding a
  decode slot / KV-cache blocks forever (generating far longer than
  expected, or the cache too full to ever schedule it). Checked once per
  engine-thread iteration against `Request.arrival_time` (already a field
  on `Request`, just not set by `LLMEngine.add_request` itself --
  `EngineWorker` sets it right after `add_request()` returns). Pass
  `timeout_s: null` in the request body to opt a request out of the
  backstop entirely.
- **Client disconnect** -- the harder one for a generation server:
  `app.py`'s SSE loop polls `raw_request.is_disconnected()` on the same
  cadence it polls for new tokens (wrapping `stream()`'s `__anext__()` in
  `asyncio.wait_for`, so neither check can be starved by waiting on the
  other) and calls `worker.abort()` the moment it notices, rather than
  discovering it only after the next token happens to be ready. A
  `CancelledError` from the ASGI server tearing the connection down out
  from under the handler gets the same treatment.

## Running it

```bash
pip install fastapi uvicorn httpx   # httpx only needed for tests (FastAPI's TestClient)
python -m server --device cuda      # needs a real GPU -- see __main__.py's module docstring
```

```bash
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": [1, 2, 3], "max_tokens": 32, "stream": true}'
```

## What's not here

- **No tokenizer.** See `schemas.py`'s module docstring -- `prompt` and
  `token_ids` are token ids throughout; a real deployment needs a tokenizer
  in front of this (encode) and behind it (decode), same gap
  `model/llm_engine.py`'s `generate()` already flags.
- **No auth, no rate limiting, no per-API-key quotas.** Out of scope for
  "wire an OpenAI-shaped API onto the engine"; would sit in front of
  `app.py`'s routes (or behind a reverse proxy) without touching
  `EngineWorker`.
- **No `/v1/chat/completions`, no multiple `n` completions per prompt, no
  `logprobs`.** `/v1/completions` with one completion per request is the
  minimum that proves the queuing/streaming/timeout plumbing end to end;
  each of these is an additive extension of `schemas.py` + `app.py`, not a
  redesign of `engine_worker.py`.
- **Batch-of-prompts-per-request.** `LLMEngine.generate()` accepts a list of
  prompts and lets the scheduler interleave them; this API only accepts one
  prompt per HTTP request (real concurrency here comes from *multiple HTTP
  requests* sharing one `EngineWorker`, which the scheduler still
  interleaves exactly the same way).
