"""Measures speculative decoding's actual speedup on the real Llama-3.2-1B
draft / Llama-3-8B target pair: acceptance rate, mean accepted tokens per
target forward pass, wall-clock inter-token latency (ITL), and tokens/s --
swept over num_speculative_tokens in {0 (non-speculative baseline), 1, 2,
4, 8}.

Unlike benchmarks/chunked_prefill/measure_itl.py and benchmarks/
prefix_caching/measure_ttft.py, this one needs real weights, not
model/minimal_llama.py's llama3_8b_shape() random init -- those only care
about scheduling/memory/compute-shape effects, but acceptance rate is
content-dependent: a random draft/target pair has ~0% correlation (same
reasoning as generate_tokenized_prompts.py's module docstring), which
would make every number here meaningless. Loads both real checkpoints via
model/hf_loader.py's load_hf_checkpoint and reuses prompts_tokenized.jsonl
(this directory, built by generate_tokenized_prompts.py) -- its own
max_tokens is overridden by --max-tokens below, since that file's cap
(24) exists for a fast correctness check, not a throughput reading.

Real-weight loading dominates cost here, unlike the toy-shaped
benchmarks: the target's weights are loaded once and reused across the
whole sweep, but LLMEngine's speculative_config path reloads the draft's
1B weights fresh every k>0 sweep point (no caching exists for this, and
adding one means touching LLMEngine.__init__ again for a benchmark-only
concern -- not worth it this late). --repeats defaults to 1 (no stdev),
a deliberate departure from this project's usual repeats-with-first-
discarded rigor (measure_itl.py/measure_ttft.py) -- that convention was
calibrated for cheap, toy-shaped runs; repeating a real-weight reload
plus generation loop several times per sweep point would multiply an
already-long run. The flag stays available if more rigor is wanted.

IMPORTANT -- unverified end to end without a GPU, same caveat as
measure_itl.py: load_prompts()/summarize_run()/aggregate_repeats() have
no CUDA dependency and can be sanity-checked anywhere; main()'s actual
engine-driving path needs a real GPU with the checkpoints on disk. Two
environment gotchas from this project's own Stage F session, still
apply: run via `sudo` (real checkpoints live under /root, only reachable
that way -- sudo resets $HOME to /root, fixing both permission and path
resolution), and `sudo docker stop vllm-server` first (it auto-starts on
boot and holds ~42GB of the L40S's ~44GB otherwise).

Smoke-test with a tiny sweep first:
    sudo .venv/bin/python3 -m benchmarks.speculative_decoding.measure_speedup \\
        --num-speculative-tokens-values 0,2 --max-tokens 8 --repeats 1

Real run:
    sudo .venv/bin/python3 -m benchmarks.speculative_decoding.measure_speedup
"""
import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

NUM_SPECULATIVE_TOKENS_VALUES = [0, 1, 2, 4, 8]  # 0 == non-speculative baseline
MAX_TOKENS = 80
BLOCK_SIZE = 16
REPEATS = 1  # see module docstring on why this defaults lower than the other benchmarks

PROMPTS_PATH = Path(__file__).parent / "prompts_tokenized.jsonl"
SUMMARY_CSV = Path(__file__).parent / "accept_summary.csv"
RAW_CSV = Path(__file__).parent / "accept_raw.csv"
STEP_CSV = Path(__file__).parent / "step_latency.csv"

# Same glob-by-repo-dir-name resolution as model/tests/test_hf_loader.py's
# _find_snapshot_dir and benchmarks/speculative_decoding's other scripts.
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


def load_prompts(path: Path, max_tokens: int) -> list:
    """Reads prompts_tokenized.jsonl, overriding every record's own
    max_tokens with the (typically much larger) value this script needs
    for a meaningful throughput reading -- no CUDA dependency, testable
    anywhere.
    """
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append({**r, "max_tokens": max_tokens})
    return records


def run_speculative_workload(engine, prompts: list) -> tuple:
    """Submits every prompt as its own request (mirrors benchmarks/
    speculative_decoding/verify_speculative_correctness.py's run_to_
    completion) and steps until all are done, instrumenting each step()
    call the same way measure_itl.py's run_mixed_workload does (per-
    request per-token latency), plus a speculative-specific accept/
    reject observer.

    Returns (itl_records, accept_records, step_records, total_time,
    total_output_tokens).

    accept_records: one dict per individual verify round (a request with
    draft_token_ids going into a step() call) -- {"request_id",
    "num_proposed", "num_accepted_total", "num_draft_accepted"}.
    num_accepted_total is how many tokens actually got committed this
    round (1..K+1, the speedup-relevant number -- includes the bonus
    token). num_draft_accepted is how many of the K *proposed* tokens
    were right (0..K, excludes the bonus token -- the standard
    literature "acceptance rate" denominator's numerator). Re-derived
    here from public Request attributes only, the same way LLMEngine.
    step()'s own internal correction block does it -- no engine changes.
    draft_token_ids must be read *before* step(), not after: by the time
    step() returns, it's already been cleared and refilled for the next
    round.
    """
    from engine.request import SamplingParams

    requests = {
        r["id"]: engine.add_request(r["prompt"], sampling_params=SamplingParams(max_tokens=r["max_tokens"]),
                                     request_id=r["id"])
        for r in prompts
    }
    t0 = time.perf_counter()
    last_token_time = {rid: t0 for rid in requests}
    prev_len = {rid: 0 for rid in requests}
    itl_records, accept_records, step_records = [], [], []
    step_index = 0

    while engine.scheduler.has_unfinished_requests():
        pre_spec = {
            r.request_id: (len(r.output_token_ids), len(r.draft_token_ids))
            for r in engine.scheduler.running if r.draft_token_ids
        }
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

        for rid, request in requests.items():
            if len(request.output_token_ids) > prev_len[rid]:
                itl_records.append({
                    "request_id": rid,
                    "token_index": len(request.output_token_ids) - 1,
                    "itl_seconds": now - last_token_time[rid],
                })
                last_token_time[rid] = now
                prev_len[rid] = len(request.output_token_ids)

            if rid in pre_spec:
                pre_len, num_proposed = pre_spec[rid]
                num_accepted_total = len(request.output_token_ids) - pre_len
                accept_records.append({
                    "request_id": rid,
                    "num_proposed": num_proposed,
                    "num_accepted_total": num_accepted_total,
                    "num_draft_accepted": min(num_accepted_total, num_proposed),
                })

    total_time = time.perf_counter() - t0
    # From this function's own `requests` dict, not engine.scheduler.
    # requests -- Scheduler.free_finished_requests() deletes an entry
    # from its own dict once a request finishes (engine/scheduler.py:359),
    # so by the time every request is done, that dict is empty. The
    # Request objects themselves are unaffected (still hold every
    # committed token) since `requests` here holds the same references
    # add_request() returned, independent of the scheduler's own map.
    total_output_tokens = sum(len(r.output_token_ids) for r in requests.values())
    return itl_records, accept_records, step_records, total_time, total_output_tokens


def summarize_run(num_speculative_tokens: int, itl_records: list, accept_records: list,
                   total_time: float, total_output_tokens: int) -> dict:
    """One repeat's arithmetic, kept separate from aggregate_repeats so
    each is independently testable -- mirrors measure_itl.py's
    summarize_itl/aggregate_repeats split.
    """
    itl_ms = [r["itl_seconds"] * 1000 for r in itl_records]
    total_proposed = sum(r["num_proposed"] for r in accept_records)
    total_draft_accepted = sum(r["num_draft_accepted"] for r in accept_records)
    total_accepted_total = sum(r["num_accepted_total"] for r in accept_records)
    num_rounds = len(accept_records)
    return {
        "num_speculative_tokens": num_speculative_tokens,
        "acceptance_rate": (total_draft_accepted / total_proposed) if total_proposed else 0.0,
        "mean_accepted_tokens_per_round": (total_accepted_total / num_rounds) if num_rounds else 1.0,
        "itl_ms_mean": statistics.mean(itl_ms) if itl_ms else 0.0,
        "tokens_per_second": total_output_tokens / total_time if total_time > 0 else 0.0,
    }


def aggregate_repeats(num_speculative_tokens: int, repeat_summaries: list) -> dict:
    """Mean +/- stdev across --repeats runs (stdev 0.0 for a single
    repeat -- same convention measure_itl.py/measure_ttft.py use).
    Target forward passes per output token is derived here, not in
    summarize_run, since it's just mean_accepted_tokens_per_round's
    reciprocal -- no point computing it twice per repeat.
    """
    def mean_and_stdev(key):
        values = [s[key] for s in repeat_summaries]
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    accept_mean, accept_stdev = mean_and_stdev("acceptance_rate")
    per_round_mean, per_round_stdev = mean_and_stdev("mean_accepted_tokens_per_round")
    itl_mean, itl_stdev = mean_and_stdev("itl_ms_mean")
    tps_mean, tps_stdev = mean_and_stdev("tokens_per_second")
    return {
        "num_speculative_tokens": num_speculative_tokens,
        "num_repeats": len(repeat_summaries),
        "acceptance_rate_mean": accept_mean,
        "acceptance_rate_stdev": accept_stdev,
        "mean_accepted_tokens_per_round_mean": per_round_mean,
        "mean_accepted_tokens_per_round_stdev": per_round_stdev,
        "target_forward_passes_per_output_token": (1 / per_round_mean) if per_round_mean else 1.0,
        "itl_ms_mean": itl_mean,
        "itl_ms_stdev": itl_stdev,
        "tokens_per_second_mean": tps_mean,
        "tokens_per_second_stdev": tps_stdev,
    }


def write_results(summary_rows: list, raw_rows: list, step_rows: list) -> None:
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_speculative_tokens", "num_repeats",
            "acceptance_rate_mean", "acceptance_rate_stdev",
            "mean_accepted_tokens_per_round_mean", "mean_accepted_tokens_per_round_stdev",
            "target_forward_passes_per_output_token",
            "itl_ms_mean", "itl_ms_stdev",
            "tokens_per_second_mean", "tokens_per_second_stdev",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_speculative_tokens", "repeat_index", "request_id",
            "num_proposed", "num_accepted_total", "num_draft_accepted",
        ])
        writer.writeheader()
        writer.writerows(raw_rows)

    with STEP_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_speculative_tokens", "repeat_index", "step_index", "duration_ms",
            "num_scheduled_tokens", "num_waiting", "num_running",
        ])
        writer.writeheader()
        writer.writerows(step_rows)

    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"Wrote {len(raw_rows)} raw rows to {RAW_CSV}")
    print(f"Wrote {len(step_rows)} step rows to {STEP_CSV}")


def _make_engine(target_config, target_weights, draft_dir, num_speculative_tokens,
                  cache_num_gpu_blocks, draft_num_gpu_blocks, block_size, max_num_seqs):
    # Deferred imports: this whole function needs a real CUDA GPU -- see
    # module docstring. Keeps load_prompts/summarize_run/aggregate_repeats
    # importable and testable on any machine.
    from engine.config import CacheConfig, SchedulerConfig, SpeculativeConfig
    from model.llm_engine import LLMEngine

    cache_config = CacheConfig(block_size=block_size, num_gpu_blocks=cache_num_gpu_blocks)
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=2048)
    spec_config = None
    if num_speculative_tokens > 0:
        spec_config = SpeculativeConfig(draft_model_path=draft_dir, num_speculative_tokens=num_speculative_tokens,
                                         draft_num_gpu_blocks=draft_num_gpu_blocks)
    return LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights,
                      device="cuda", speculative_config=spec_config)


def warmup(k_values: list, prompts: list, draft_dir, target_config, target_weights,
           cache_num_gpu_blocks, draft_num_gpu_blocks, block_size) -> None:
    """One small untimed run per sweep point before real measurement --
    same rationale as measure_itl.py's warmup(): match the real batch
    shape (same prompts, same requests), only shrink what's safe to
    shrink (here: nothing needs shrinking, the prompt set is already
    small -- this just primes Triton autotuning/CUDA kernel caches for
    every k value before the timed runs).
    """
    t0 = time.perf_counter()
    tiny_prompts = [{**p, "max_tokens": 4} for p in prompts]
    for k in k_values:
        engine = _make_engine(target_config, target_weights, draft_dir, k,
                               cache_num_gpu_blocks, draft_num_gpu_blocks, block_size, len(prompts))
        run_speculative_workload(engine, tiny_prompts)
    print(f"Warmup done ({len(k_values)} sweep points) in {time.perf_counter() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-speculative-tokens-values", default=",".join(str(n) for n in NUM_SPECULATIVE_TOKENS_VALUES),
                     help="Comma-separated K values to sweep; 0 means the non-speculative baseline")
    ap.add_argument("--max-tokens", default=MAX_TOKENS, type=int,
                     help="Overrides prompts_tokenized.jsonl's own (much smaller) per-prompt cap")
    ap.add_argument("--block-size", default=BLOCK_SIZE, type=int)
    ap.add_argument("--num-gpu-blocks", default=None, type=int,
                     help="Target's KV cache; defaults to exact-fit for all prompts at --max-tokens, +1 block margin")
    ap.add_argument("--draft-num-gpu-blocks", default=None, type=int,
                     help="Draft's own (much smaller) KV cache; same default sizing as --num-gpu-blocks")
    ap.add_argument("--repeats", default=REPEATS, type=int,
                     help="Measurements per sweep point, averaged (with stdev) -- see module docstring on why "
                          "this defaults lower than the other benchmarks' REPEATS")
    ap.add_argument("--keep-first-repeat", action="store_true",
                     help="Include repeat 0 instead of discarding it as an extra warmup (see measure_ttft.py)")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    k_values = [int(n) for n in args.num_speculative_tokens_values.split(",")]

    target_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    draft_dir = _find_snapshot_dir("models--meta-llama--Llama-3.2-1B-Instruct")
    if target_dir is None or draft_dir is None:
        raise SystemExit(
            f"Real checkpoints not found under {_HF_HUB_DIR} -- "
            "see ~/.claude/plans/agile-rolling-gray.md's Context section."
        )

    from model.hf_loader import load_hf_checkpoint
    target_config, target_weights = load_hf_checkpoint(target_dir, device="cuda")

    prompts = load_prompts(PROMPTS_PATH, args.max_tokens)
    if not prompts:
        raise SystemExit(
            f"No prompts found at {PROMPTS_PATH} -- run "
            "python3 -m benchmarks.speculative_decoding.generate_tokenized_prompts first."
        )

    num_gpu_blocks = args.num_gpu_blocks
    if num_gpu_blocks is None:
        num_gpu_blocks = sum(-(-(len(p["prompt"]) + p["max_tokens"]) // args.block_size) for p in prompts) + 1
    draft_num_gpu_blocks = args.draft_num_gpu_blocks or num_gpu_blocks

    if not args.skip_warmup:
        warmup(k_values, prompts, draft_dir, target_config, target_weights,
               num_gpu_blocks, draft_num_gpu_blocks, args.block_size)

    discard_first_repeat = not args.keep_first_repeat
    total_repeats = args.repeats + 1 if discard_first_repeat else args.repeats

    summary_rows, raw_rows, step_rows = [], [], []
    for k in k_values:
        repeat_summaries = []
        for repeat_index in range(total_repeats):
            engine = _make_engine(target_config, target_weights, draft_dir, k,
                                   num_gpu_blocks, draft_num_gpu_blocks, args.block_size, len(prompts))
            itl_records, accept_records, step_records, total_time, total_output_tokens = \
                run_speculative_workload(engine, prompts)

            repeat_summaries.append(summarize_run(k, itl_records, accept_records, total_time, total_output_tokens))
            for rec in accept_records:
                raw_rows.append({"num_speculative_tokens": k, "repeat_index": repeat_index, **rec})
            for step in step_records:
                step_rows.append({"num_speculative_tokens": k, "repeat_index": repeat_index, **step})

        kept_summaries = repeat_summaries[1:] if discard_first_repeat else repeat_summaries
        aggregated = aggregate_repeats(k, kept_summaries)
        summary_rows.append(aggregated)
        print(f"num_speculative_tokens={k}: {aggregated}")

    write_results(summary_rows, raw_rows, step_rows)


if __name__ == "__main__":
    main()
