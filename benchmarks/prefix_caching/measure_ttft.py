"""Measures TTFT (time to first token) with prefix caching on vs. off, for a
workload of requests sharing a common system-prompt prefix of varying
length -- the scenario engine/prefix_cache.py's RadixTrie (wired into
BlockManager/Scheduler, see engine/README.md's "Prefix caching" section) is
meant to speed up.

Drives model.llm_engine.LLMEngine directly, one CacheConfig.
enable_prefix_caching=False/True pair per prefix length in PREFIX_LENS, and
writes two CSVs (see write_results' docstring) -- turning those into a "TTFT
reduction % vs. prefix length" chart is a separate step, through this
repo's usual dataviz/artifact tooling, not this script's job.

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

    python3 benchmarks/prefix_caching/measure_ttft.py \\
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


def run_workload(engine, prompts: list, max_tokens: int = 1) -> dict:
    """Submits every prompt at once (one shared t0, modeling concurrent
    arrival) and drives engine.step() directly rather than
    LLMEngine.generate() -- generate() doesn't expose per-step timing, and
    TTFT is exactly the thing being measured here. Returns
    {request_id: ttft_seconds}.

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
    prev_len = {r.request_id: 0 for r in requests}
    while engine.scheduler.has_unfinished_requests():
        engine.step()
        now = time.perf_counter()
        for r in requests:
            if r.request_id not in ttft and len(r.output_token_ids) > prev_len[r.request_id]:
                ttft[r.request_id] = now - t0
            prev_len[r.request_id] = len(r.output_token_ids)
    return ttft


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


def write_results(summary_rows: list, raw_rows: list) -> None:
    """summary: prefix_len, cache_off_ttft_ms, cache_on_ttft_ms,
    reduction_pct -- one row per prefix length swept. raw: prefix_len,
    cache_enabled, request_id, ttft_ms -- every individual measurement,
    for a richer chart later than the summary alone supports.
    """
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix_len", "cache_off_ttft_ms", "cache_on_ttft_ms", "reduction_pct"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix_len", "cache_enabled", "request_id", "ttft_ms"])
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"Wrote {len(raw_rows)} raw rows to {RAW_CSV}")


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

    summary_rows, raw_rows = [], []
    for prefix_len in prefix_lens:
        prompts = build_workload(prefix_len, args.num_requests, args.suffix_len, seed=args.seed)

        torch.manual_seed(args.seed)
        engine_off = _make_engine(model_config, weights, num_gpu_blocks, args.block_size,
                                   enable_prefix_caching=False, max_num_seqs=args.num_requests)
        ttft_off = run_workload(engine_off, prompts, max_tokens=args.max_tokens)

        torch.manual_seed(args.seed)
        engine_on = _make_engine(model_config, weights, num_gpu_blocks, args.block_size,
                                  enable_prefix_caching=True, max_num_seqs=args.num_requests)
        ttft_on = run_workload(engine_on, prompts, max_tokens=args.max_tokens)

        summary_rows.append(summarize_results(prefix_len, ttft_off, ttft_on))
        for request_id, t in ttft_off.items():
            raw_rows.append({"prefix_len": prefix_len, "cache_enabled": False, "request_id": request_id, "ttft_ms": t * 1000})
        for request_id, t in ttft_on.items():
            raw_rows.append({"prefix_len": prefix_len, "cache_enabled": True, "request_id": request_id, "ttft_ms": t * 1000})

        print(f"prefix_len={prefix_len}: {summary_rows[-1]}")

    write_results(summary_rows, raw_rows)


if __name__ == "__main__":
    main()