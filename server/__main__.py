"""`python -m server` -- the one CUDA-only file in this package: constructs
a real model/llm_engine.py LLMEngine and serves it behind server/app.py's
FastAPI app with uvicorn.

Not imported by anything else in server/ (app.py and engine_worker.py both
take an already-built worker/engine so their tests don't need this file at
all -- see their module docstrings), so this being import-time CUDA/Triton-
dependent (LLMEngine -> model/kv_cache.py -> model/minimal_llama.py ->
kernels/flash_attention.py's `import triton`) doesn't block running the rest
of this package's tests on a machine without a GPU. Written and statically
reviewed on such a machine; not yet run for real -- next step is the L40S,
same as model/'s own CUDA-only pieces.
"""
import argparse

import uvicorn

from engine.config import CacheConfig, SchedulerConfig
from model.llm_engine import LLMEngine
from model.minimal_llama import llama3_8b_shape
from server.app import create_app
from server.engine_worker import EngineWorker


def build_worker(args: argparse.Namespace) -> EngineWorker:
    model_config = llama3_8b_shape(n_layers=args.n_layers, max_seq_len=args.max_model_len)
    cache_config = CacheConfig(block_size=args.block_size, num_gpu_blocks=args.num_gpu_blocks)
    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
    )
    engine = LLMEngine(
        cache_config, scheduler_config, model_config,
        max_model_len=args.max_model_len, device=args.device, seed=args.seed,
    )
    return EngineWorker(engine, default_timeout_s=args.default_timeout_s)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # llama3_8b_shape's own defaults (n_layers=4, max_seq_len=128) are a
    # deliberately truncated toy shape -- see model/minimal_llama.py's
    # module docstring -- not sized for real generation; override both for
    # anything beyond smoke-testing the wiring.
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--default-timeout-s", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    worker = build_worker(args)
    app = create_app(worker)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
