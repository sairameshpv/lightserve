"""CUDA Graph capture of one autoregressive decode step over
model/minimal_llama.py, benchmarked against the same decode loop run eager
(no graph) -- the launch-overhead story CUDA graphs exist for is sharpest
exactly here: batch=1 decode does very little GPU work per kernel (one
token's worth), so the CPU time spent *issuing* each of the dozens of
kernel launches per forward call (matmuls, RMSNorm, FlashAttention, RoPE's
handful of elementwise ops) can dominate over the GPU time spent actually
running them. `torch.cuda.graph()` replaces that whole per-step launch
sequence with one `graph.replay()` call.

Static-shape decode, on purpose: real incremental-KV-cache decode (query =
1 new token, key/value = a *growing* cache) needs kernels/flash_attention.py's
FA kernel to accept Nq != Nkv, which it doesn't (see that file's forward
signature -- q.shape == k.shape == v.shape is asserted). CUDA graphs also
require identical tensor shapes on every replay. Both constraints are
satisfied at once the way current production stacks satisfy the *second*
one too (e.g. HF's StaticCache + torch.compile(mode="reduce-overhead")):
keep a single fixed-size [B, max_seq_len] input_ids buffer, and every
decode step reruns the *full* forward pass over it (causal, unchanged
FlashAttention kernel, unchanged shape every call). Only the newest token
is ever written into the buffer; everything from position `cur_len` onward
is still the initial fill value (a valid-but-meaningless token id, never
0-reset mid-run) -- causal masking guarantees row `cur_len - 1`'s output
(the one actually read each step) can never be contaminated by those
not-yet-real rows, since causal attention only lets a row read columns
<= itself. The real cost of this choice is recomputing O(max_seq_len^2)
self-attention every step instead of O(cur_len) -- a real inefficiency vs.
an incremental cache, traded here for reusing the existing FA kernel
unmodified and keeping the CUDA-graph mechanics the focus of this file.
Kept `--max-seq-len` modest (128 by default) so that trade doesn't dominate
the measurement.

Usage (run as a module, not a script, so `model.minimal_llama`'s absolute
import resolves the same way pytest's `-m pytest model/tests/` already
does -- `python3 model/cuda_graph_decode.py` puts model/ itself, not the
repo root, on sys.path and fails with ModuleNotFoundError: model):
    python3 -m model.cuda_graph_decode
    python3 -m model.cuda_graph_decode --n-layers 8 --num-steps 100
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from model.minimal_llama import init_weights, llama3_8b_shape, llama_forward

RESULTS_FILE = Path(__file__).parent / "cuda_graph_decode_results.json"
_SEED_TOKEN_ID = 1


def _make_seeded_buffer(batch_size, max_seq_len, device):
    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    input_ids[:, 0] = _SEED_TOKEN_ID  # position 0 = the one real token every run starts from
    return input_ids


def _count_cuda_kernel_launches(weights, config, batch_size):
    """How many discrete CUDA kernels one forward call issues -- this is
    the number `graph.replay()` collapses to 1. Measured with
    torch.profiler, not hand-counted, same "verify on real hardware, don't
    hand-estimate" habit as kernels/README.md's shared-memory config probe.
    Best-effort: profiler internals vary across torch versions, so this
    returns None (and the rest of the script still runs) rather than
    crashing the benchmark over a secondary measurement.
    """
    input_ids = _make_seeded_buffer(batch_size, config.max_seq_len, "cuda")
    # Warm up *before* profiling: the first call on a given shape also pays
    # every matmul/attention kernel's one-time @triton.autotune search
    # (itself many extra kernel launches, benchmarking several tile-size
    # candidates), which is cached and skipped on every call after -- a
    # cold call would count that one-time cost as if every decode step
    # paid it, which it doesn't. Steady state is the number that matters.
    llama_forward(weights, config, input_ids, causal=True)
    torch.cuda.synchronize()
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            llama_forward(weights, config, input_ids, causal=True)
            torch.cuda.synchronize()
        return sum(1 for e in prof.events() if e.device_type == torch.profiler.DeviceType.CUDA)
    except Exception as e:  # noqa: BLE001 -- best-effort, see docstring
        print(f"  (kernel-launch count unavailable: {e})")
        return None


def _time_decode_loop(step_fn, num_steps, warmup_steps):
    """step_fn() runs exactly one decode step. Returns post-warmup per-step
    latencies in ms, CUDA-event-timed (captures real GPU-timeline elapsed
    time, including any idle gaps from the CPU being slow to issue the next
    launch -- exactly the effect being measured, not just kernel runtime).
    """
    for _ in range(warmup_steps):
        step_fn()
    torch.cuda.synchronize()

    latencies = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(num_steps):
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return latencies


def run_eager(weights, config, batch_size, num_steps, warmup_steps):
    input_ids = _make_seeded_buffer(batch_size, config.max_seq_len, "cuda")
    state = {"cur_len": 1}

    def step():
        logits = llama_forward(weights, config, input_ids, causal=True)
        next_tok = logits[:, state["cur_len"] - 1, :].argmax(dim=-1)
        input_ids[:, state["cur_len"]] = next_tok
        state["cur_len"] += 1

    return _time_decode_loop(step, num_steps, warmup_steps)


def run_cuda_graph(weights, config, batch_size, num_steps, warmup_steps):
    input_ids = _make_seeded_buffer(batch_size, config.max_seq_len, "cuda")

    # Side-stream warmup before capture: required by torch.cuda.graph's own
    # contract, and also what forces every kernel here's @triton.autotune
    # search (kernels 3 and 4's) to run and cache its winning config *now*
    # -- autotune's internal benchmarking does real CPU-GPU syncs, which are
    # illegal inside torch.cuda.graph()'s capture region.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            llama_forward(weights, config, input_ids, causal=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_logits = llama_forward(weights, config, input_ids, causal=True)

    state = {"cur_len": 1}

    def step():
        graph.replay()
        next_tok = static_logits[:, state["cur_len"] - 1, :].argmax(dim=-1)
        input_ids[:, state["cur_len"]] = next_tok
        state["cur_len"] += 1

    return _time_decode_loop(step, num_steps, warmup_steps)


def _summarize(latencies):
    s = sorted(latencies)
    return {
        "median_ms": statistics.median(s),
        "mean_ms": statistics.mean(s),
        "p95_ms": s[int(0.95 * (len(s) - 1))],
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-layers", type=int, default=4, help="real LLaMA-3-8B has 32 -- see module docstring")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1, help="1 is where launch overhead matters most")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"

    config = llama3_8b_shape(n_layers=args.n_layers, max_seq_len=args.max_seq_len)
    # Every decode step writes one more real token and advances cur_len;
    # this file always reruns the full max_seq_len buffer (see module
    # docstring), so each run needs enough buffer room for its own
    # warmup+timed steps, plus cuda graph capture's separate 3-iteration
    # warmup (started fresh from cur_len=1 too).
    needed = args.warmup_steps + args.num_steps + 3
    assert needed < config.max_seq_len, (
        f"--max-seq-len {config.max_seq_len} too small for --warmup-steps {args.warmup_steps} + "
        f"--num-steps {args.num_steps} (need room for {needed} decode steps < max_seq_len)"
    )

    print(f"Config: n_layers={config.n_layers} hidden={config.hidden_size} n_heads={config.n_heads} "
          f"head_dim={config.head_dim} intermediate={config.intermediate_size} vocab={config.vocab_size} "
          f"max_seq_len={config.max_seq_len} batch_size={args.batch_size} dtype={config.dtype}")

    weights = init_weights(config, device="cuda", seed=args.seed)

    print("\nCounting CUDA kernel launches per forward call...")
    launch_count = _count_cuda_kernel_launches(weights, config, args.batch_size)
    if launch_count is not None:
        print(f"  {launch_count} CUDA kernels per forward call (n_layers={config.n_layers}) "
              f"-> 1 graph.replay() call")

    print("\nEager (no CUDA graph)...")
    eager_latencies = run_eager(weights, config, args.batch_size, args.num_steps, args.warmup_steps)

    print("CUDA graph (captured once, replayed per step)...")
    graph_latencies = run_cuda_graph(weights, config, args.batch_size, args.num_steps, args.warmup_steps)

    eager_stats = _summarize(eager_latencies)
    graph_stats = _summarize(graph_latencies)
    speedup = eager_stats["median_ms"] / graph_stats["median_ms"] if graph_stats["median_ms"] > 0 else float("nan")

    print()
    print(f"{'mode':<12} {'median (ms)':>12} {'mean (ms)':>12} {'p95 (ms)':>10}")
    print(f"{'eager':<12} {eager_stats['median_ms']:>12.3f} {eager_stats['mean_ms']:>12.3f} {eager_stats['p95_ms']:>10.3f}")
    print(f"{'cuda_graph':<12} {graph_stats['median_ms']:>12.3f} {graph_stats['mean_ms']:>12.3f} {graph_stats['p95_ms']:>10.3f}")
    print(f"\nSpeedup (median eager / median cuda_graph): {speedup:.2f}x")

    results = {
        "config": {
            "n_layers": config.n_layers, "hidden_size": config.hidden_size, "n_heads": config.n_heads,
            "head_dim": config.head_dim, "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size, "max_seq_len": config.max_seq_len,
            "batch_size": args.batch_size, "dtype": str(config.dtype),
        },
        "launch_count_per_forward": launch_count,
        "eager": {**eager_stats, "latencies_ms": eager_latencies},
        "cuda_graph": {**graph_stats, "latencies_ms": graph_latencies},
        "speedup_median": speedup,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
