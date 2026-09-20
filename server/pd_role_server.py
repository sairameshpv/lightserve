"""Minimal, standalone HTTP role-server for the real 2-GPU P/D
disaggregation benchmark (benchmarks/pd_disaggregation/
measure_pd_real_speedup.py). One running process is either a PREFILL node
or a DECODE node, never both, built directly on model/pd_disaggregation.py's
existing prefill()/resume_decode() primitives -- no engine/ changes.

Deliberately does NOT extend server/engine_worker.py, even though the
decode role needs the exact same "one background thread owns the engine,
everything else goes through a thread-safe queue" pattern EngineWorker
already solved (see its own module docstring for the full reasoning --
LLMEngine.step() must never run on the event loop, or every other
in-flight request stalls for that step's duration). EngineWorker's
submission path always calls engine.add_request() with fresh prompt
tokens; this role only ever needs to resume_decode() an already-prefilled
handoff and only needs the final result, not per-token streaming -- a real
subset of EngineWorker's job. Duplicating the (small) thread-safety
plumbing here, as _DecodeWorker below, keeps engine_worker.py's own
tested, production server/app.py path completely untouched -- same
"reuse the pattern, not the file" precedent model/draft_proposer.py set
for its shadow-Request design.

The prefill role does NOT need this machinery at all: in this benchmark's
own workload (see measure_pd_real_speedup.py), the prefill node only ever
handles one request at a time during the timed measurement window (the
"already decoding" population's own earlier prefills happen in an
untimed setup phase; the one injected long prefill is alone on that node
during the timed window) -- so a plain `async def` handler calling the
blocking prefill() function directly, letting concurrent requests simply
queue behind each other at the ASGI level, is correct and far simpler
than adding a second background worker for a case that never arises here.

Wire format: JSON everywhere except where a request/response actually
carries tensors (POST /prefill's response when a handoff results; POST
/resume_decode's request body) -- those use torch.save/torch.load over
raw bytes (weights_only=False: both ends are this project's own trusted
code on its own two boxes, carrying plain dicts/dataclasses alongside the
tensors, not just tensors -- torch 2.6's weights_only=True default is a
hardening for loading untrusted checkpoints, which doesn't apply here).
Chosen over hand-rolled struct+JSON framing because torch.save already
does exactly this (serialize an arbitrary Python object, tensors
included, to bytes) with no numpy dependency and far less code. Chosen
over base64/multipart because both ends are code we control and neither
adds anything but overhead. Chosen over real RDMA because Nebius's L40S
platform (gpu-l40s-a) doesn't expose InfiniBand fabric at all -- checked
against Nebius's own docs before picking this (only the SXM-class
platforms -- H100/H200/B200/B300 -- support fabric clusters); plain
networking is the only real transport option here, not a compromise.

Run manually, one process per node:
    # on the prefill node:
    sudo .venv/bin/python3 -m server.pd_role_server --role prefill --port 8100 --num-gpu-blocks N
    # on the decode node:
    sudo .venv/bin/python3 -m server.pd_role_server --role decode --port 8100 --num-gpu-blocks N
"""
import argparse
import io
import os
import queue
import threading
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import Response

from engine.config import CacheConfig, SchedulerConfig
from engine.request import SamplingParams
from model.pd_disaggregation import PrefillHandoff, prefill, resume_decode

# Same glob-by-repo-dir-name resolution as every other real-checkpoint
# script in this repo -- inlined rather than imported, same reasoning
# each of those gives.
_HF_HUB_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def _find_snapshot_dir(model_repo_dir_name):
    snapshots_dir = os.path.join(_HF_HUB_DIR, model_repo_dir_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for name in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, name)
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


def pack(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def unpack(body: bytes):
    return torch.load(io.BytesIO(body), weights_only=False)


class _DecodeWorker:
    """Background-thread-owns-the-engine pattern -- see module docstring
    on why this duplicates (a small slice of) server/engine_worker.py's
    EngineWorker rather than extending it.
    """

    def __init__(self, engine, idle_poll_s: float = 0.005):
        self.engine = engine
        self.idle_poll_s = idle_poll_s
        self._incoming: "queue.SimpleQueue" = queue.SimpleQueue()
        self._pending: dict = {}  # request_id -> (Request, done_queue), engine-thread-only
        self._prev_len: dict = {}  # request_id -> last-seen len(output_token_ids), engine-thread-only
        # One {"request_id", "token_index", "t"} dict per token actually
        # generated, appended to only by the engine thread -- lets a
        # client observe decode ITL for requests still mid-flight (e.g.
        # "disrupted by a concurrent prefill on a different node")
        # without needing real per-token HTTP streaming: post-hoc,
        # exactly the same summarize_itl()-style gap computation
        # benchmarks/chunked_prefill/measure_itl.py already does
        # in-process, just fed from GET /itl_log instead. See
        # reset_itl_log() for why a caller clears this before a timed
        # measurement window.
        self.itl_log: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)

    def submit(self, handoff: PrefillHandoff) -> "queue.SimpleQueue":
        """Called from any thread. Returns the done-queue immediately --
        resume_decode() itself (which touches engine.scheduler directly)
        still only ever runs on the engine thread, same reasoning
        EngineWorker.submit()'s own docstring gives for deferring
        add_request the same way.
        """
        done_q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._incoming.put((handoff, done_q))
        return done_q

    def reset_itl_log(self) -> None:
        """Clears the log -- called (via POST /reset_itl_log) right
        before a timed measurement window starts, so an untimed setup
        phase's own token growth (e.g. settling a decode workload before
        injecting a disruptive prefill elsewhere) doesn't pollute it.
        Thread-safe: itl_log is only ever appended to by the engine
        thread, and list.clear() is atomic under the GIL against a
        concurrent append -- no lock needed, same reasoning this whole
        class already leans on for _pending.
        """
        self.itl_log.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain_incoming()
            if self.engine.scheduler.has_unfinished_requests():
                self.engine.step()
                self._record_itl()
                self._check_finished()
            else:
                time.sleep(self.idle_poll_s)

    def _drain_incoming(self) -> None:
        while True:
            try:
                handoff, done_q = self._incoming.get_nowait()
            except queue.Empty:
                return
            request = resume_decode(self.engine, handoff)
            self._pending[request.request_id] = (request, done_q)
            # Seeded at the request's current length (1 -- the first
            # output token resume_decode() already carried over from
            # prefill, see model/pd_disaggregation.py), not 0 -- so
            # _record_itl only ever logs tokens genuinely generated on
            # *this* engine, never that already-carried-over one.
            self._prev_len[request.request_id] = len(request.output_token_ids)

    def _record_itl(self) -> None:
        now = time.time()
        for rid, (request, _) in self._pending.items():
            prev = self._prev_len[rid]
            cur = len(request.output_token_ids)
            for token_index in range(prev, cur):
                self.itl_log.append({"request_id": rid, "token_index": token_index, "t": now})
            self._prev_len[rid] = cur

    def _check_finished(self) -> None:
        # request.status is already updated by the step() call just above
        # (execute_model's own maybe_finish() call, same timing
        # model/pd_disaggregation.py's own generate_disaggregated() relies
        # on) -- this is purely observing that, not driving it.
        finished_ids = [rid for rid, (req, _) in self._pending.items() if req.is_finished()]
        for rid in finished_ids:
            request, done_q = self._pending.pop(rid)
            self._prev_len.pop(rid, None)
            done_q.put((request.output_token_ids, request.status.name))


def create_prefill_app(engine) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "role": "prefill"}

    @app.post("/prefill")
    async def do_prefill(raw: FastAPIRequest):
        body = await raw.json()
        sampling_params = SamplingParams(max_tokens=body["max_tokens"], eos_token_id=body.get("eos_token_id"))
        t0 = time.perf_counter()
        # Blocking call, deliberately not backgrounded -- see module
        # docstring on why concurrent prefill traffic never actually
        # arises in this benchmark's own workload.
        result = prefill(engine, body["prompt_token_ids"], sampling_params=sampling_params,
                          request_id=body.get("request_id"))
        prefill_seconds = time.perf_counter() - t0
        return Response(content=pack({"result": result, "prefill_seconds": prefill_seconds}),
                         media_type="application/octet-stream")

    return app


def create_decode_app(engine) -> FastAPI:
    worker = _DecodeWorker(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "role": "decode"}

    @app.post("/resume_decode")
    async def do_resume_decode(raw: FastAPIRequest):
        body = await raw.body()
        handoff = unpack(body)
        done_q = worker.submit(handoff)
        # asyncio.to_thread, not a plain blocking .get(): multiple
        # concurrent /resume_decode requests must all be able to wait
        # simultaneously without stalling the shared event loop from
        # accepting/submitting each other's requests in the meantime --
        # the whole point of this role handling several requests at once.
        import asyncio
        output_token_ids, finish_reason = await asyncio.to_thread(done_q.get)
        return {"request_id": handoff.request_id, "output_token_ids": output_token_ids,
                "finish_reason": finish_reason}

    @app.post("/reset_itl_log")
    def reset_itl_log():
        worker.reset_itl_log()
        return {"status": "ok"}

    @app.get("/itl_log")
    def get_itl_log():
        return {"records": worker.itl_log}

    return app


def build_engine(args: argparse.Namespace):
    # Deferred import: this whole function needs a real CUDA GPU -- keeps
    # module-level imports here limited to what argument parsing/wire
    # framing need, matching every other real-checkpoint script's own
    # "torch/fastapi are fine at module level, the engine build isn't"
    # split in this repo.
    from model.hf_loader import load_hf_checkpoint
    from model.llm_engine import LLMEngine

    checkpoint_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    if checkpoint_dir is None:
        raise SystemExit(
            f"Real Llama-3-8B-Instruct checkpoint not found under {_HF_HUB_DIR} -- "
            "see ~/.claude/plans/agile-rolling-gray.md's Context section."
        )
    model_config, weights = load_hf_checkpoint(checkpoint_dir, device="cuda")
    cache_config = CacheConfig(block_size=args.block_size, num_gpu_blocks=args.num_gpu_blocks,
                                int8_kv=args.int8_kv)
    scheduler_config = SchedulerConfig(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=2048)
    return LLMEngine(cache_config, scheduler_config, model_config, weights=weights, device="cuda")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", required=True, choices=["prefill", "decode"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, required=True,
                         help="No universal default -- sizing depends on the specific benchmark "
                              "workload the caller knows about, same as this repo's other real-"
                              "checkpoint scripts' own --num-gpu-blocks handling.")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--int8-kv", action="store_true",
                         help="Store this engine's KV cache at int8 instead of the checkpoint's "
                              "own dtype (see engine/config.py's CacheConfig.int8_kv). Off by "
                              "default -- prefill and decode nodes choose this independently, "
                              "each on their own command line.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    engine = build_engine(args)
    app = create_prefill_app(engine) if args.role == "prefill" else create_decode_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
