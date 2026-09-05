"""Measures TTFT (time to first token) with prefix caching on vs. off, for a
workload of requests sharing a common system-prompt prefix of varying
length -- the scenario engine/prefix_cache.py's RadixTrie (wired into
BlockManager/Scheduler, see engine/README.md's "Prefix caching" section) is
meant to speed up.

Drives model.llm_engine.LLMEngine directly, one CacheConfig.
enable_prefix_caching=False/True pair per prefix length in PREFIX_LENS
(each pair preceded by an untimed warmup() pass -- see its docstring for
why: without it, whatever one-time CUDA/Triton kernel-compile cost exists
lands entirely on cache_off's numbers, since it always runs first), and
writes three CSVs (see write_results' docstring) -- turning those into a
"TTFT reduction % vs. prefix length" chart is a separate step, through
this repo's usual dataviz/artifact tooling, not this script's job.

This repo has no tokenizer (see benchmarks/generate_token_prompts.py's
module docstring for the same point made about baseline_prompts.jsonl), so
the workload is synthetic random token ids, matching that script's own
VOCAB_SIZE convention.

IMPORTANT -- unverified end to end on this machine: LLMEngine/ModelRunner/
init_weights all hard-require a real CUDA GPU (`device="cuda"`, no CPU
fallback anywhere in model/ -- see model/tests/test_llm_engine.py's
requires_cuda-gated tests), so main()'s actual engine-driving path has only
been reviewed, never run, on a machine without one. build_workload() and
summarize_results() below have no such dependency and ARE covered by
benchmarks/tests/test_measure_ttft.py, runnable anywhere. Before trusting a
full sweep, smoke-test on the real GPU first with a tiny one, e.g.:

    python3 -m benchmarks.prefix_caching.measure_ttft \\
        --prefix-lens 0,32 --num-requests 4 --max-tokens 1

(same spirit as run_baseline.py's own --limit flag for exactly this reason).
"""
import argparse
import csv
import random
import time
from pathlib import Path

# meta-llama/Meta-Llama-3-8B-Instruct's real vocab size -- matches
# benchmarks/generate_token_prompts.py's VOCAB_SIZE and
# model/minimal_llama.py's llama3_8b_shape() default.
VOCAB_SIZE = 128_256

PREFIX_LENS = [0, 128, 256, 512, 1024, 2048]  # whole multiples of the default block_size=16
NUM_REQUESTS = 32
SUFFIX_LEN = 16

SUMMARY_CSV = Path(__file__).parent / "prefix_cache_ttft_summary.csv"
RAW_CSV = Path(__file__).parent / "prefix_cache_ttft_raw.csv"
STEP_CSV = Path(__file__).parent / "prefix_cache_step_latency.csv"


def build_workload(prefix_len: int, num_requests: int, suffix_len: int,
                    vocab_size: int = VOCAB_SIZE, seed: int = 0) -> list:
    """num_requests prompts, each `shared_prefix + a unique random suffix`
    -- shared_prefix is identical (same content, same length) across every
    request in the returned list, modeling one workload of concurrent
    requests all carrying the same system prompt.
    """
    rng = random.Random(seed)
    shared_prefix = [rng.randrange(vocab_size) for _ in range(prefix_len)]
    return [
        shared_prefix + [rng.randrange(vocab_size) for _ in range(suffix_len)]
        for _ in range(num_requests)
    ]


def run_workload(engine, prompts: list, max_tokens: int = 1) -> tuple:
    """Submits every prompt at once (one shared t0, modeling concurrent
    arrival) and drives engine.step() directly rather than
    LLMEngine.generate() -- generate() doesn't expose per-step timing, and
    TTFT is exactly the thing being measured here. Returns
    ({request_id: ttft_seconds}, step_records) -- step_records is one dict
    per engine.step() call (see its keys below), for diagnosing *why* a
    configuration is slow, not just that it is.

    Caveat, stated plainly rather than left implicit: one timestamp is
    captured per whole batched step(), not per request within a step, so a
    request's TTFT here is "when the batch containing its first output
    token finished", not a GPU-event-level per-request timestamp. Fine for
    a relative reduction-% comparison across cache on/off; not a claim of
    sub-step precision.
    """
    from engine.request import SamplingParams  # local import -- see module docstring

    t0 = time.perf_counter()
    requests = [
        engine.add_request(p, sampling_params=SamplingParams(max_tokens=max_tokens))
        for p in prompts
    ]
    ttft = {}
    step_records = []
    prev_len = {r.request_id: 0 for r in requests}
    step_index = 0
    while engine.scheduler.has_unfinished_requests():
        step_start = time.perf_counter()
        output, _ = engine.step()
        now = time.perf_counter()
        step_records.append({
            "step_index": step_index,
            "duration_ms": (now - step_start) * 1000,
            "num_scheduled_tokens": output.total_num_scheduled_tokens,
            "num_waiting": len(engine.scheduler.waiting),
            "num_running": len(engine.scheduler.running),
        })
        step_index += 1
        for r in requests:
            if r.request_id not in ttft and len(r.output_token_ids) > prev_len[r.request_id]:
                ttft[r.request_id] = now - t0
            prev_len[r.request_id] = len(r.output_token_ids)
    return ttft, step_records


def summarize_results(prefix_len: int, ttft_off: dict, ttft_on: dict) -> dict:
    """Mean TTFT (across all requests in the workload) for one prefix
    length, both cache states, plus the reduction percentage. Kept as a
    separate, torch-free function so its arithmetic is unit-testable
    without an engine at all (see benchmarks/tests/test_measure_ttft.py).
    """
    mean_off = sum(ttft_off.values()) / len(ttft_off)
    mean_on = sum(ttft_on.values()) / len(ttft_on)
    reduction_pct = 100 * (mean_off - mean_on) / mean_off if mean_off > 0 else 0.0
    return {
        "prefix_len": prefix_len,
        "cache_off_ttft_ms": mean_off * 1000,
        "cache_on_ttft_ms": mean_on * 1000,
        "reduction_pct": reduction_pct,
    }


def write_results(summary_rows: list, raw_rows: list, step_rows: list) -> None:
    """summary: prefix_len, cache_off_ttft_ms, cache_on_ttft_ms,
    reduction_pct -- one row per prefix length swept. raw: prefix_len,
    cache_enabled, request_id, ttft_ms -- every individual measurement,
    for a richer chart later than the summary alone supports. step:
    prefix_len, cache_enabled, step_index, duration_ms,
    num_scheduled_tokens, num_waiting, num_running -- one row per
    engine.step() call (see run_workload's docstring), for diagnosing
    *why* a configuration is slow rather than just that it is.
    """
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix_len", "cache_off_ttft_ms", "cache_on_ttft_ms", "reduction_pct"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix_len", "cache_enabled", "request_id", "ttft_ms"])
        writer.writeheader()
        writer.writerows(raw_rows)

    with STEP_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prefix_len", "cache_enabled", "step_index", "duration_ms",
            "num_scheduled_tokens", "num_waiting", "num_running",
        ])
        writer.writeheader()
        writer.writerows(step_rows)

    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"Wrote {len(raw_rows)} raw rows to {RAW_CSV}")
    print(f"Wrote {len(step_rows)} step rows to {STEP_CSV}")


def warmup(prefix_lens: list, suffix_len: int, block_size: int, num_gpu_blocks: int,
           model_config, weights, seed: int, num_requests: int = 4) -> None:
    """Runs one small, untimed workload per prefix length in `prefix_lens`,
    both cache off and on, through throwaway engines, before any real
    (timed) measurement.

    Why: main()'s sweep always builds and times `engine_off` before
    `engine_on`, for every prefix length -- whatever one-time CUDA context
    / Triton kernel-compile cost exists for a given call shape would
    otherwise land entirely on that prefix length's `cache_off` number,
    not because caching did anything, just because `off` happened to run
    first while the shape was still cold.

    Both cache states are run here, not just one: a first cut of this
    function only ran caching off, on the (wrong) assumption that "warm
    shared Triton kernel compilation" only depends on shapes like
    ordinary dense prefill/decode -- but model/model_runner.py's
    _attention() pads every non-fresh-prefill call (`L < se`, see its
    docstring) up to `se` regardless of how many rows are real, and a
    cache-*hit* continuation's exact (se, num-real-rows) combination
    never occurs on a cache-off run at all (there, prefill always
    completes in the same step it starts a request's very first chunk, so
    a decode step's se is prefill-length-plus-one, one more than a cache
    hit's prefill-length-plus-chunk -- close, but not the same call, and
    apparently close enough to matter to Triton's autotuner). Measured
    consequence: every prefix length except one paid a 1.3-2 *second*
    one-time cost on the real sweep's first cache-hit continuation step,
    dwarfing every other step in that same run (structurally-identical
    later steps cost ~11-20ms) -- see benchmarks/prefix_caching/README.md.

    The cache-on pass submits a donor *first*, through its own separate
    run_workload() call, and waits for it to fully finish (so its prefix
    is genuinely registered into the cache -- see
    BlockManager.insert_computed_prefix) before submitting `num_requests`
    followers sharing its prefix in a second call. Deliberately not "just
    submit num_requests requests sharing a prefix in one run_workload()
    call, same as the real sweep does": a second cut of this function did
    exactly that and still left prefix_lens 128/256/512 unwarmed, only
    fixing 1024/2048 by accident -- a small enough `num_requests` at a
    short enough `prefix_len` lets every toy request's tokens fit inside
    one scheduler step's token budget, so (exactly the mechanism behind
    the ref-count bug this whole feature already hit once, see
    engine/prefix_cache.py's git history) *all* of them get admitted
    before *any* of them has registered anything -- never producing a
    genuine matched continuation at all for those lengths. Splitting into
    two separate run_workload() calls forces the donor's registration to
    complete first, regardless of prefix_len or budget sizing.

    Uses a small fixed `num_requests` (4) rather than the real sweep's
    --num-requests, to keep this cheap -- covers per-token kernel shapes
    but not necessarily batch-size-specific compiled variants. A
    reasonable cost/rigor trade-off; revisit if results still look
    warmup-biased.
    """
    t0 = time.perf_counter()
    for prefix_len in prefix_lens:
        off_prompts = build_workload(prefix_len, num_requests, suffix_len, seed=seed)
        engine_off = _make_engine(model_config, weights, num_gpu_blocks, block_size,
                                   enable_prefix_caching=False, max_num_seqs=num_requests)
        run_workload(engine_off, off_prompts, max_tokens=1)

        engine_on = _make_engine(model_config, weights, num_gpu_blocks, block_size,
                                  enable_prefix_caching=True, max_num_seqs=num_requests)
        # build_workload restarts its own rng fresh from `seed` every call,
        # so a donor drawn with n=1 and followers drawn with n=num_requests
        # would both start with the exact same first suffix -- follower[0]
        # would be a byte-for-byte duplicate of the donor (100% matched,
        # chunk=0), which model/model_runner.py's _attention doesn't
        # handle (`[-L:]` with L=0 is `[0:]` in Python, not empty -- see
        # this function's git history for the crash that surfaced this).
        # Drawing num_requests+1 and dropping the first (the donor's own
        # duplicate) keeps every real follower's suffix from ever colliding
        # with the donor's, without touching that separate, real bug.
        donor_prompts = build_workload(prefix_len, 1, suffix_len, seed=seed)
        run_workload(engine_on, donor_prompts, max_tokens=1)  # runs to completion -> prefix registered
        follower_prompts = build_workload(prefix_len, num_requests + 1, suffix_len, seed=seed)[1:]
        run_workload(engine_on, follower_prompts, max_tokens=1)  # genuine matched continuation, guaranteed
    print(f"Warmup done ({len(prefix_lens)} shapes x 2 cache states) in {time.perf_counter() - t0:.1f}s")


def _make_engine(model_config, weights, num_gpu_blocks, block_size, enable_prefix_caching, max_num_seqs):
    # Deferred imports: this whole function needs a real CUDA GPU (see
    # module docstring) -- keeping torch/model/engine.block_manager out of
    # this module's top level lets build_workload/summarize_results/
    # write_results stay importable and testable on any machine.
    from engine.config import CacheConfig, SchedulerConfig
    from model.llm_engine import LLMEngine

    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks,
                                enable_prefix_caching=enable_prefix_caching)
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=4096)
    return LLMEngine(cache_config, scheduler_config, model_config, weights=weights, device="cuda")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix-lens", default=",".join(str(n) for n in PREFIX_LENS),
                     help="Comma-separated shared-prefix lengths to sweep")
    ap.add_argument("--num-requests", default=NUM_REQUESTS, type=int)
    ap.add_argument("--suffix-len", default=SUFFIX_LEN, type=int)
    ap.add_argument("--max-tokens", default=1, type=int, help="Output tokens per request -- kept small on purpose")
    ap.add_argument("--block-size", default=16, type=int)
    ap.add_argument("--num-gpu-blocks", default=None, type=int,
                     help="Defaults to enough for the worst case (caching off, longest prefix)")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--skip-warmup", action="store_true",
                     help="Skip the untimed warmup pass -- faster iteration on the harness itself, "
                          "but real measurements will be biased by cold-start (see warmup()'s docstring)")
    args = ap.parse_args()

    prefix_lens = [int(n) for n in args.prefix_lens.split(",")]

    import torch
    from dataclasses import replace
    from model.minimal_llama import init_weights, llama3_8b_shape

    max_len_needed = max(prefix_lens) + args.suffix_len + args.max_tokens
    model_config = llama3_8b_shape(max_seq_len=max_len_needed)
    weights = init_weights(model_config, device="cuda", seed=args.seed)

    num_gpu_blocks = args.num_gpu_blocks
    if num_gpu_blocks is None:
        blocks_per_request = -(-max_len_needed // args.block_size)  # ceil div
        num_gpu_blocks = blocks_per_request * args.num_requests + args.block_size  # +1 block margin

    if not args.skip_warmup:
        warmup(prefix_lens, args.suffix_len, args.block_size, num_gpu_blocks,
               model_config, weights, args.seed)

    summary_rows, raw_rows, step_rows = [], [], []
    for prefix_len in prefix_lens:
        prompts = build_workload(prefix_len, args.num_requests, args.suffix_len, seed=args.seed)

        torch.manual_seed(args.seed)
        engine_off = _make_engine(model_config, weights, num_gpu_blocks, args.block_size,
                                   enable_prefix_caching=False, max_num_seqs=args.num_requests)
        ttft_off, steps_off = run_workload(engine_off, prompts, max_tokens=args.max_tokens)

        torch.manual_seed(args.seed)
        engine_on = _make_engine(model_config, weights, num_gpu_blocks, args.block_size,
                                  enable_prefix_caching=True, max_num_seqs=args.num_requests)
        ttft_on, steps_on = run_workload(engine_on, prompts, max_tokens=args.max_tokens)

        summary_rows.append(summarize_results(prefix_len, ttft_off, ttft_on))
        for request_id, t in ttft_off.items():
            raw_rows.append({"prefix_len": prefix_len, "cache_enabled": False, "request_id": request_id, "ttft_ms": t * 1000})
        for request_id, t in ttft_on.items():
            raw_rows.append({"prefix_len": prefix_len, "cache_enabled": True, "request_id": request_id, "ttft_ms": t * 1000})
        for cache_enabled, steps in ((False, steps_off), (True, steps_on)):
            for step in steps:
                step_rows.append({"prefix_len": prefix_len, "cache_enabled": cache_enabled, **step})

        print(f"prefix_len={prefix_len}: {summary_rows[-1]}")

    write_results(summary_rows, raw_rows, step_rows)


if __name__ == "__main__":
    main()