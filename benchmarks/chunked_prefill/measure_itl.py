"""Measures decode inter-token latency (ITL) when a long prefill request
arrives into an already-decoding workload, at several chunk sizes -- the
scenario Scheduler's chunked-prefill admission (see engine/README.md's
"Chunked prefill" section) exists for: without it, one long prefill can
monopolize a whole step's wall-clock time, stalling every other request's
next token for that entire step.

The mechanism, worth stating precisely because it shapes what this script
actually measures: Scheduler._schedule_running always services every
already-running decode request's 1-token need *before* _schedule_waiting
admits anything new, and a chunked-prefill continuation is serviced
through that same _schedule_running loop, *after* the decode requests
already in it. So a decode request's token is never dropped from a step's
schedule by chunk size -- what varies is how long that step itself takes
in wall-clock time. An unchunked (single-shot) long prefill makes one
step's forward pass huge, so every decode request's next token arrives
however long that one giant step took; chunking spreads the same total
prefill work over many small steps, each close to a normal decode step's
duration. ITL is therefore a direct read on step wall-clock time, not on
scheduling fairness.

Drives model.llm_engine.LLMEngine directly, one SchedulerConfig.
max_num_batched_tokens (the chunk size) per value in --chunk-sizes,
repeated --repeats times each and averaged (mean +/- stdev, first repeat
discarded by default -- see measure_ttft.py's own git history for why
this exists: a single measurement of a GPU workload is noisy enough to
get conclusions wrong, and warmup can't be trusted to anticipate every
shape in advance). See warmup()'s docstring for the one lesson carried
over from that file's saga that matters here.

This repo has no tokenizer (see benchmarks/generate_token_prompts.py's
module docstring), so the workload is synthetic random token ids.

IMPORTANT -- unverified end to end on this machine: LLMEngine/ModelRunner/
init_weights all hard-require a real CUDA GPU (see
model/tests/test_llm_engine.py's requires_cuda-gated tests), so main()'s
actual engine-driving path has only been reviewed, never run, on a
machine without one. build_decode_prompts()/build_prefill_prompt()/
summarize_itl()/aggregate_repeats() below have no such dependency and ARE
covered by benchmarks/tests/test_measure_itl.py, runnable anywhere.
Before trusting a full sweep, smoke-test on the real GPU first with a
tiny one:

    python3 -m benchmarks.chunked_prefill.measure_itl \\
        --num-decode-requests 2 --decode-max-tokens 5 --settle-steps 2 \\
        --prefill-len 256 --chunk-sizes 128,1024 --repeats 1
"""
import argparse
import csv
import random
import statistics
import time
from pathlib import Path

# meta-llama/Meta-Llama-3-8B-Instruct's real vocab size -- matches
# benchmarks/generate_token_prompts.py's VOCAB_SIZE and
# model/minimal_llama.py's llama3_8b_shape() default.
VOCAB_SIZE = 128_256

NUM_DECODE_REQUESTS = 8
DECODE_PROMPT_LEN = 16
DECODE_MAX_TOKENS = 100
SETTLE_STEPS = 10          # steps run before the long prefill is injected
PREFILL_LEN = 2048
PREFILL_MAX_TOKENS = 1
CHUNK_SIZES = [128, 256, 512, 1024, 2048, 8192]  # 8192 > PREFILL_LEN -> effectively unchunked
REPEATS = 5                # see benchmarks/prefix_caching/measure_ttft.py's REPEATS comment

SUMMARY_CSV = Path(__file__).parent / "itl_summary.csv"
RAW_CSV = Path(__file__).parent / "itl_raw.csv"
STEP_CSV = Path(__file__).parent / "step_latency.csv"


def build_decode_prompts(num_requests: int, prompt_len: int,
                          vocab_size: int = VOCAB_SIZE, seed: int = 0) -> list:
    """num_requests independent short prompts -- these model an
    already-busy decode workload, not a shared-prefix one (see
    benchmarks/prefix_caching/measure_ttft.py's build_workload for that
    shape), so every prompt is its own fresh random draw.
    """
    rng = random.Random(seed)
    return [[rng.randrange(vocab_size) for _ in range(prompt_len)] for _ in range(num_requests)]


def build_prefill_prompt(prefill_len: int, vocab_size: int = VOCAB_SIZE, seed: int = 1) -> list:
    """One long prompt -- the request that arrives mid-stream and disrupts
    the decode workload above. Default seed differs from
    build_decode_prompts' so the two never accidentally draw the same
    content from the same rng state.
    """
    rng = random.Random(seed)
    return [rng.randrange(vocab_size) for _ in range(prefill_len)]


def run_mixed_workload(engine, decode_prompts: list, decode_max_tokens: int,
                        prefill_prompt: list, prefill_max_tokens: int, settle_steps: int) -> tuple:
    """Submits decode_prompts at t0, runs settle_steps steps so they reach
    steady-state decode, *then* submits prefill_prompt (the long request)
    and keeps stepping until everything finishes.

    Returns (itl_records, step_records, prefill_ttft_seconds).

    itl_records: one dict per consecutive-token gap, per decode request --
    {"request_id", "token_index", "phase", "itl_seconds"}. `phase` is
    "baseline" for gaps entirely before the prefill was injected, or
    "disrupted" for the first gap that spans the injection point onward --
    exactly the split that shows whether chunking kept decode smooth
    through the disruption or not. The prefill request's own tokens are
    not included here (see PREFILL_MAX_TOKENS -- it's not the subject of
    this measurement, it's the disruption).

    step_records: one dict per engine.step() call, same shape as
    measure_ttft.py's run_workload (duration_ms, num_scheduled_tokens,
    num_waiting, num_running) -- lets a chunk size's effect be seen
    directly in step timing, not just inferred from ITL.
    """
    from engine.request import SamplingParams  # local import -- see module docstring

    t0 = time.perf_counter()
    decode_requests = [
        engine.add_request(p, sampling_params=SamplingParams(max_tokens=decode_max_tokens))
        for p in decode_prompts
    ]
    last_token_time = {r.request_id: t0 for r in decode_requests}
    prev_len = {r.request_id: 0 for r in decode_requests}
    itl_records = []
    step_records = []
    step_index = 0
    # Sentinels the prefill request doesn't have a value for until it's
    # actually injected below -- run_step (called during the settle-steps
    # loop, before that happens) reads these guarded by the None check.
    prefill_t0 = None
    prefill_request = None
    prefill_ttft_holder = [None]  # a list so run_step's closure can write into it

    def run_step(phase_if_disrupted):
        nonlocal step_index
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
        for r in decode_requests:
            if len(r.output_token_ids) > prev_len[r.request_id]:
                itl_records.append({
                    "request_id": r.request_id,
                    "token_index": len(r.output_token_ids) - 1,
                    "phase": phase_if_disrupted,
                    "itl_seconds": now - last_token_time[r.request_id],
                })
                last_token_time[r.request_id] = now
                prev_len[r.request_id] = len(r.output_token_ids)
        if prefill_request is not None and prefill_ttft_holder[0] is None and prefill_request.output_token_ids:
            prefill_ttft_holder[0] = now - prefill_t0
        return now

    for _ in range(settle_steps):
        run_step("baseline")

    prefill_t0 = time.perf_counter()
    prefill_request = engine.add_request(
        prefill_prompt, sampling_params=SamplingParams(max_tokens=prefill_max_tokens)
    )

    while engine.scheduler.has_unfinished_requests():
        run_step("disrupted")

    return itl_records, step_records, prefill_ttft_holder[0]


def summarize_itl(chunk_size: int, itl_records: list, prefill_ttft: float) -> dict:
    """One repeat's arithmetic -- kept separate from aggregate_repeats so
    each is independently testable (see benchmarks/tests/test_measure_itl.py),
    mirroring measure_ttft.py's summarize_results/aggregate_repeats split.
    """
    baseline = [r["itl_seconds"] for r in itl_records if r["phase"] == "baseline"]
    disrupted = [r["itl_seconds"] for r in itl_records if r["phase"] == "disrupted"]
    return {
        "chunk_size": chunk_size,
        "baseline_itl_ms_mean": statistics.mean(baseline) * 1000 if baseline else 0.0,
        "disrupted_itl_ms_mean": statistics.mean(disrupted) * 1000 if disrupted else 0.0,
        "disrupted_itl_ms_max": max(disrupted) * 1000 if disrupted else 0.0,
        "prefill_ttft_ms": prefill_ttft * 1000 if prefill_ttft is not None else 0.0,
    }


def aggregate_repeats(chunk_size: int, repeat_summaries: list) -> dict:
    """Mean +/- stdev across --repeats runs' worth of summarize_itl()
    dicts, all for the same chunk_size. stdev is 0.0 for a single repeat
    (see measure_ttft.py's aggregate_repeats for the same convention).
    """
    def mean_and_stdev(key):
        values = [s[key] for s in repeat_summaries]
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    baseline_mean, baseline_stdev = mean_and_stdev("baseline_itl_ms_mean")
    disrupted_mean, disrupted_stdev = mean_and_stdev("disrupted_itl_ms_mean")
    disrupted_max_mean, disrupted_max_stdev = mean_and_stdev("disrupted_itl_ms_max")
    ttft_mean, ttft_stdev = mean_and_stdev("prefill_ttft_ms")
    return {
        "chunk_size": chunk_size,
        "num_repeats": len(repeat_summaries),
        "baseline_itl_ms_mean": baseline_mean,
        "baseline_itl_ms_stdev": baseline_stdev,
        "disrupted_itl_ms_mean": disrupted_mean,
        "disrupted_itl_ms_stdev": disrupted_stdev,
        "disrupted_itl_ms_max_mean": disrupted_max_mean,
        "disrupted_itl_ms_max_stdev": disrupted_max_stdev,
        "prefill_ttft_ms_mean": ttft_mean,
        "prefill_ttft_ms_stdev": ttft_stdev,
    }


def write_results(summary_rows: list, raw_rows: list, step_rows: list) -> None:
    """summary: chunk_size, num_repeats, baseline/disrupted ITL mean+stdev,
    disrupted ITL max (mean+stdev of each repeat's own max), prefill TTFT
    mean+stdev -- one row per chunk size swept. raw: chunk_size,
    repeat_index, request_id, token_index, phase, itl_ms -- every
    individual gap, every repeat. step: chunk_size, repeat_index,
    step_index, duration_ms, num_scheduled_tokens, num_waiting,
    num_running -- lets the chunk-size-vs-step-duration relationship this
    whole benchmark is built around be inspected directly.
    """
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chunk_size", "num_repeats",
            "baseline_itl_ms_mean", "baseline_itl_ms_stdev",
            "disrupted_itl_ms_mean", "disrupted_itl_ms_stdev",
            "disrupted_itl_ms_max_mean", "disrupted_itl_ms_max_stdev",
            "prefill_ttft_ms_mean", "prefill_ttft_ms_stdev",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chunk_size", "repeat_index", "request_id", "token_index", "phase", "itl_ms",
        ])
        writer.writeheader()
        writer.writerows(raw_rows)

    with STEP_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "chunk_size", "repeat_index", "step_index", "duration_ms",
            "num_scheduled_tokens", "num_waiting", "num_running",
        ])
        writer.writeheader()
        writer.writerows(step_rows)

    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"Wrote {len(raw_rows)} raw rows to {RAW_CSV}")
    print(f"Wrote {len(step_rows)} step rows to {STEP_CSV}")


def warmup(chunk_sizes: list, num_decode_requests: int, decode_prompt_len: int, prefill_len: int,
           block_size: int, num_gpu_blocks: int, model_config, weights, seed: int) -> None:
    """Runs one small, untimed run_mixed_workload per chunk size, before
    any real (timed) measurement.

    The one lesson carried over from measure_ttft.py's warmup() saga
    (five iterations there before the real cause -- Triton's
    @triton.autotune keying on the flat batch size M -- was found and
    fixed for good, see its git history): match the *batch size* real
    measurements will use, don't shrink it for cheapness. Here that means
    `num_decode_requests` and `prefill_len` stay exactly what the real
    sweep uses -- only `settle_steps` and `decode_max_tokens` shrink,
    since those just need to be *long enough* to touch each step shape
    once (pure decode, and decode-plus-chunk at this chunk_size), not
    representative of the real run's full duration.
    """
    t0 = time.perf_counter()
    decode_prompts = build_decode_prompts(num_decode_requests, decode_prompt_len, seed=seed)
    prefill_prompt = build_prefill_prompt(prefill_len, seed=seed + 1)
    for chunk_size in chunk_sizes:
        engine = _make_engine(model_config, weights, num_gpu_blocks, block_size,
                               chunk_size, num_decode_requests + 1)
        run_mixed_workload(engine, decode_prompts, decode_max_tokens=3,
                            prefill_prompt=prefill_prompt, prefill_max_tokens=1, settle_steps=2)
    print(f"Warmup done ({len(chunk_sizes)} chunk sizes) in {time.perf_counter() - t0:.1f}s")


def _make_engine(model_config, weights, num_gpu_blocks, block_size, chunk_size, max_num_seqs):
    # Deferred imports: this whole function needs a real CUDA GPU (see
    # module docstring) -- keeping torch/model/engine.block_manager out of
    # this module's top level lets the pure-Python helpers above stay
    # importable and testable on any machine.
    from engine.config import CacheConfig, SchedulerConfig
    from model.llm_engine import LLMEngine

    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=chunk_size)
    return LLMEngine(cache_config, scheduler_config, model_config, weights=weights, device="cuda")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-decode-requests", default=NUM_DECODE_REQUESTS, type=int,
                     help="Concurrent already-decoding requests the long prefill arrives into")
    ap.add_argument("--decode-prompt-len", default=DECODE_PROMPT_LEN, type=int)
    ap.add_argument("--decode-max-tokens", default=DECODE_MAX_TOKENS, type=int,
                     help="Output tokens per decode request -- how many ITL samples each contributes")
    ap.add_argument("--settle-steps", default=SETTLE_STEPS, type=int,
                     help="Steps run before the long prefill is injected, for a clean baseline ITL")
    ap.add_argument("--prefill-len", default=PREFILL_LEN, type=int,
                     help="Length of the one long prompt injected mid-stream")
    ap.add_argument("--prefill-max-tokens", default=PREFILL_MAX_TOKENS, type=int)
    ap.add_argument("--chunk-sizes", default=",".join(str(n) for n in CHUNK_SIZES),
                     help="Comma-separated SchedulerConfig.max_num_batched_tokens values to sweep "
                          "-- a value >= --prefill-len is effectively unchunked")
    ap.add_argument("--block-size", default=16, type=int)
    ap.add_argument("--num-gpu-blocks", default=None, type=int,
                     help="Defaults to enough for all decode requests plus the long prefill, "
                          "sized once regardless of chunk size (chunk size doesn't change memory needs)")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--repeats", default=REPEATS, type=int,
                     help="Measurements per chunk size, averaged (with stdev) -- see REPEATS' comment")
    ap.add_argument("--keep-first-repeat", action="store_true",
                     help="Include repeat 0 in the mean/stdev instead of discarding it as an extra "
                          "warmup (see measure_ttft.py's --keep-first-repeat for why this defaults off)")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    chunk_sizes = [int(n) for n in args.chunk_sizes.split(",")]

    from model.minimal_llama import init_weights, llama3_8b_shape

    max_len_needed = args.prefill_len + args.prefill_max_tokens
    model_config = llama3_8b_shape(max_seq_len=max_len_needed)
    weights = init_weights(model_config, device="cuda", seed=args.seed)

    num_gpu_blocks = args.num_gpu_blocks
    if num_gpu_blocks is None:
        decode_len = args.decode_prompt_len + args.decode_max_tokens
        decode_blocks = -(-decode_len // args.block_size) * args.num_decode_requests
        prefill_blocks = -(-max_len_needed // args.block_size)
        num_gpu_blocks = decode_blocks + prefill_blocks + args.block_size  # +1 block margin

    if not args.skip_warmup:
        warmup(chunk_sizes, args.num_decode_requests, args.decode_prompt_len, args.prefill_len,
               args.block_size, num_gpu_blocks, model_config, weights, args.seed)

    discard_first_repeat = not args.keep_first_repeat
    total_repeats = args.repeats + 1 if discard_first_repeat else args.repeats

    summary_rows, raw_rows, step_rows = [], [], []
    for chunk_size in chunk_sizes:
        # Same prompts (same seed) every repeat, deliberately -- see
        # measure_ttft.py's identical comment on why.
        decode_prompts = build_decode_prompts(args.num_decode_requests, args.decode_prompt_len, seed=args.seed)
        prefill_prompt = build_prefill_prompt(args.prefill_len, seed=args.seed + 1)

        repeat_summaries = []
        for repeat_index in range(total_repeats):
            engine = _make_engine(model_config, weights, num_gpu_blocks, args.block_size,
                                   chunk_size, args.num_decode_requests + 1)
            itl_records, step_records, prefill_ttft = run_mixed_workload(
                engine, decode_prompts, args.decode_max_tokens,
                prefill_prompt, args.prefill_max_tokens, args.settle_steps,
            )

            repeat_summaries.append(summarize_itl(chunk_size, itl_records, prefill_ttft))
            for rec in itl_records:
                raw_rows.append({
                    "chunk_size": chunk_size, "repeat_index": repeat_index,
                    "request_id": rec["request_id"], "token_index": rec["token_index"],
                    "phase": rec["phase"], "itl_ms": rec["itl_seconds"] * 1000,
                })
            for step in step_records:
                step_rows.append({"chunk_size": chunk_size, "repeat_index": repeat_index, **step})

        kept_summaries = repeat_summaries[1:] if discard_first_repeat else repeat_summaries
        aggregated = aggregate_repeats(chunk_size, kept_summaries)
        summary_rows.append(aggregated)
        print(f"chunk_size={chunk_size}: {aggregated}")

    write_results(summary_rows, raw_rows, step_rows)


if __name__ == "__main__":
    main()
