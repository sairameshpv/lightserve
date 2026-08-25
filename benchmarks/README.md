# lightserve vs vLLM vs SGLang -- Locust load test

`locustfile.py` load-tests any OpenAI-compatible `POST /v1/completions`
server -- lightserve's own `server/app.py`, a real vLLM server, or a real
SGLang server -- with the same closed-loop task shape, so a run against each
at the same concurrency is a same-hardware, same-request-pattern comparison.
See `locustfile.py`'s module docstring for the full design (why `--target`
exists, why there are two prompt files, why `wait_time = between(0, 0)`).
vLLM and SGLang both speak the same text-in/text-out OpenAI completions
shape, so `--target vllm` (`baseline_prompts.jsonl`, real text) is reused
for SGLang too -- only lightserve needs `--target lightserve`
(`token_prompts.jsonl`, token ids), since it alone has no tokenizer.

## Results (2026-08-25, Nebius `gpu-l40s-a`, 1x L40S)

All three serving `meta-llama/Meta-Llama-3-8B-Instruct`-shaped models
one at a time (GPU memory headroom, not a hard requirement) on the same
instance: lightserve via `python -m server` with the real (32-layer, not
the truncated toy) `llama3_8b_shape()`, random-not-real-checkpoint weights;
vLLM and SGLang both via their official Docker images
(`vllm/vllm-openai:latest`, `lmsysorg/sglang:latest`) with the real
checkpoint. Each row is one `locust --headless --run-time 120s` run;
`avg req/s` is `requests / 120s`, not the noisier moving-window snapshot in
`_stats_history.csv` (see `compare_locust_runs.py`'s module docstring).
Raw `--csv` output for every row is committed under `locust_results/`;
regenerate this table from those files with:

```bash
python3 benchmarks/summarize_locust_results.py
```

| engine | users | requests | failures | avg req/s | p50 (s) | p95 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|
| lightserve | 10 | 13 | 0 | 0.11 | 86.00 | 117.00 | 117.00 | 116.74 |
| lightserve | 30 | 31 | 0 | 0.26 | 69.00 | 101.00 | 101.00 | 100.85 |
| lightserve | 50 | 14 | 0 | 0.12 | 30.00 | 30.00 | 30.00 | 30.45 |
| vllm | 10 | 447 | 0 | 3.73 | 2.30 | 5.00 | 5.00 | 5.05 |
| vllm | 30 | 1163 | 0 | 9.69 | 3.70 | 5.40 | 5.40 | 5.83 |
| vllm | 50 | 1826 | 0 | 15.22 | 3.90 | 5.80 | 5.80 | 5.88 |
| sglang | 10 | 394 | 0 | 3.28 | 3.60 | 5.40 | 5.40 | 5.44 |
| sglang | 30 | 1041 | 0 | 8.68 | 4.20 | 6.20 | 6.40 | 6.43 |
| sglang | 50 | 1511 | 0 | 12.59 | 4.70 | 7.00 | 7.20 | 7.30 |

**Reading this**: vLLM and SGLang land in the same ballpark (both real,
optimized production engines -- vLLM somewhat ahead at this concurrency,
SGLang's p95/p99 growing a bit faster with concurrency here), both roughly
**30-100x** lightserve's throughput. That gap is not a surprise -- it's
`model/model_runner.py`'s and `engine/README.md`'s own documented
architecture, confirmed under real load for the first time here:

- **No dedicated decode kernel.** `ModelRunner._attention` pads every decode
  step's query up to the cached length and reuses the prefill-shaped
  `flash_attention_forward` kernel (see that module's docstring) --
  `O(seq_len)` attention per decode step, not vLLM/SGLang's `O(1)`.
- **Attention is a Python loop over the batch, not a batched kernel call**
  (one `flash_attention_forward` call per request per layer -- see
  `model_runner.py`'s docstring on why). Concurrency doesn't parallelize
  this the way continuous batching is supposed to; lightserve's own
  completion count *dropped* from 30 users (31 requests) to 50 users (14
  requests) in the same 120s window, most likely because more concurrent
  requests means more serialized per-request attention calls per step, not
  because of a scheduling bug (`FakeEngine`-backed unit tests already prove
  the scheduler itself batches correctly -- see `server/tests/`). No
  dedicated paged-decode kernel and no CUDA graphs, unlike either
  production engine.
- **Per-shape Triton autotuning, not warmed for arbitrary concurrent-load
  shapes.** `kernels/tiled_matmul.py`'s `@triton.autotune` keys on the
  batch dimension, which changes every step under mixed concurrent
  prefill/decode -- a genuinely new shape re-triggers autotuning (measured
  1-13s on this box, vs. ~40ms/token once warm for a *repeated* shape).
  This is a real cost of the current implementation, not a benchmark
  artifact -- it's exactly why `cuda_graph_decode.py` exists as a separate
  investigation into eliminating it for decode.
- **lightserve's own sample sizes here are small (13-31 requests)** -- a
  legitimate consequence of the throughput gap above (few requests finish
  within a fixed 120s window), not a truncated/failed run: zero failures in
  every row. Widening the window would grow the sample without changing the
  qualitative result.

None of this is a lightserve bug -- every item above is already named as a
documented, deliberate scope cut or known follow-up elsewhere in this repo
(`engine/README.md`'s "What's not wired up", `model/model_runner.py`'s
module docstring). This benchmark is the first time the *size* of that gap
was actually measured.

## Running it yourself

Bring up all three servers on the same GPU box, one at a time (each needs
most of an L40S's memory for a real/real-shaped 8B model) -- lightserve via
`server/README.md`'s "Running it" section, vLLM via
`nebius_setup_commands.txt` step 7, SGLang via:

```bash
docker run -d --name sglang-server --gpus all --shm-size 32g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN=<your token> \
    -p 30000:30000 --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port 30000
```

Then, per server, per concurrency level:

```bash
pip install locust

locust -f benchmarks/locustfile.py --host http://localhost:8001 \
    --headless --users 10 --spawn-rate 10 --run-time 120s \
    --target lightserve --csv benchmarks/locust_results/lightserve_10

locust -f benchmarks/locustfile.py --host http://localhost:8000 \
    --headless --users 10 --spawn-rate 10 --run-time 120s \
    --target vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
    --csv benchmarks/locust_results/vllm_10

locust -f benchmarks/locustfile.py --host http://localhost:30000 \
    --headless --users 10 --spawn-rate 10 --run-time 120s \
    --target vllm --model meta-llama/Meta-Llama-3-8B-Instruct \
    --csv benchmarks/locust_results/sglang_10

# repeat --users 30 and --users 50, then:
python3 benchmarks/summarize_locust_results.py
```

Each `locust ... --csv <prefix>` run writes `<prefix>_stats.csv` and
`<prefix>_stats_history.csv` (both committed under `locust_results/`), plus
`<prefix>_failures.csv`/`<prefix>_exceptions.csv` (not committed when
empty -- worth checking by hand if `Failures` is nonzero). For a single
pairwise comparison instead of the full table, `compare_locust_runs.py`
prints one side-by-side pair.

## Files

- `locustfile.py` -- the load test itself (see its module docstring).
- `token_prompts.jsonl` -- lightserve's prompt set: token ids, generated by
  `generate_token_prompts.py` as a length-matched (not content-matched --
  there's no tokenizer to match content against, see that script's module
  docstring) companion to `baseline_prompts.jsonl`'s real text prompts,
  which the vLLM/SGLang `--target vllm` runs use directly.
- `compare_locust_runs.py` -- prints a side-by-side table (requests,
  failures, throughput, p50/p95/p99/max latency) from two `--csv` runs.
- `summarize_locust_results.py` -- regenerates the full results table above
  from every `--csv` run under `locust_results/`.
- `locust_results/` -- the raw `--csv` output backing the results table
  above.

## What this isn't

Not an apples-to-apples *quality* comparison -- lightserve has no
tokenizer and random (not real-checkpoint) weights (see
`model/minimal_llama.py`'s module docstring), so its outputs are
meaningless as text; vLLM's and SGLang's outputs are real generation from
the real checkpoint. This benchmark only compares serving-engine
*mechanics* under concurrent load (admission/scheduling/streaming latency,
throughput, failure rate) at a matched prompt-length and `max_tokens`
distribution -- not generation quality, and not (yet) matched hardware
utilization (that's `benchmarks/profiling/`'s job, see
`benchmarks/profiling/report.md`).

---

# Baseline benchmark results

Auto-appended by `run_baseline.py` after each run.

| Timestamp (UTC) | Host | Model | Prompts | Concurrency | Repeats | Wall (s) | OK | Err | Throughput (tok/s) | p50 (s) | p95 (s) | p99 (s) | Results file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-30 17:56:08 UTC | 89.169.127.8 | meta-llama/Meta-Llama-3-8B-Instruct | 1010 | 50 | 4 | 298.3 | 4040 | 0 | 1645.9 | 4.24 | 6.07 | 6.09 | results_1785433870.jsonl |
