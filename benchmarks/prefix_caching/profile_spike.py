"""One-off diagnostic, not part of the regular benchmark suite: profiles
the specific cache-hit continuation step that stays slow (~1.3s) at
prefix_len 128/256/512 even after three warmup fixes in measure_ttft.py
(see its warmup() docstring and git history) -- wraps it in
torch.profiler to see what it's actually spending time on (a slow
kernel, many small kernel launches, or CUDA memory allocation) instead
of guessing at another warmup fix.

Run manually on a CUDA GPU:
    python3 -m benchmarks.prefix_caching.profile_spike --prefix-len 256
"""
import argparse

import torch

from benchmarks.prefix_caching.measure_ttft import _make_engine, build_workload, warmup
from engine.request import SamplingParams
from model.minimal_llama import init_weights, llama3_8b_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=256)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--suffix-len", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", action="store_true",
                     help="Run measure_ttft.warmup() for this exact prefix_len first, to verify "
                          "whether it actually makes the profiled step below fast.")
    args = ap.parse_args()

    max_len_needed = args.prefix_len + args.suffix_len + 1
    model_config = llama3_8b_shape(max_seq_len=max_len_needed)
    weights = init_weights(model_config, device="cuda", seed=args.seed)
    blocks_per_request = -(-max_len_needed // args.block_size)  # ceil div
    num_gpu_blocks = blocks_per_request * (args.num_requests + 1) + args.block_size

    if args.warmup:
        import time
        t0 = time.perf_counter()
        warmup([args.prefix_len], args.suffix_len, args.block_size, num_gpu_blocks,
               model_config, weights, args.seed, num_requests=args.num_requests)
        print(f"warmup() took {time.perf_counter() - t0:.1f}s")

    engine = _make_engine(model_config, weights, num_gpu_blocks, args.block_size,
                           enable_prefix_caching=True, max_num_seqs=args.num_requests + 1)

    # Donor: run to completion first, so its prefix is genuinely
    # registered before the followers below ever get admitted (same
    # donor-then-followers split measure_ttft.py's warmup() uses, and for
    # the same reason -- see its docstring on why "all admitted in one
    # run_workload() call" doesn't reliably produce a real cache hit).
    donor_prompt = build_workload(args.prefix_len, 1, args.suffix_len, seed=args.seed)[0]
    engine.add_request(donor_prompt, sampling_params=SamplingParams(max_tokens=1))
    while engine.scheduler.has_unfinished_requests():
        engine.step()
    print(f"Donor done. Cache stats: {engine.scheduler.block_manager.prefix_cache.stats()}")

    # Followers: admit them all, then profile the steps that run their
    # cache-hit continuation -- this is the part that stays slow.
    follower_prompts = build_workload(args.prefix_len, args.num_requests + 1, args.suffix_len, seed=args.seed)[1:]
    for p in follower_prompts:
        engine.add_request(p, sampling_params=SamplingParams(max_tokens=1))

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        step_idx = 0
        while engine.scheduler.has_unfinished_requests():
            with torch.profiler.record_function(f"step_{step_idx}"):
                engine.step()
            torch.cuda.synchronize()
            step_idx += 1
    print(f"\n{step_idx} steps run for {len(follower_prompts)} followers.\n")

    print("--- by CUDA time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print("\n--- by CPU time (catches launch-overhead / allocator-bound cases) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
    print("\n--- by call count (catches many-small-launches cases) ---")
    print(prof.key_averages().table(sort_by="count", row_limit=25))

    trace_path = f"/tmp/prefix_cache_spike_{args.prefix_len}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nFull trace written to {trace_path} (open in chrome://tracing or ui.perfetto.dev)")


if __name__ == "__main__":
    main()
