# Chunked prefill: decode ITL vs. chunk size

Measures decode inter-token latency (ITL) when a long prefill request
arrives into an already-decoding workload, at several chunk sizes
(`SchedulerConfig.max_num_batched_tokens`) -- the scenario chunked prefill
(see `engine/README.md`'s "Chunked prefill" section) exists for: without
it, one long prefill can monopolize a whole step's wall-clock time,
stalling every other request's next token for that entire step.

The mechanism, worth stating precisely because it shapes what this
measures: `Scheduler._schedule_running` always services every
already-running decode request's 1-token need *before* `_schedule_waiting`
admits anything new, and a chunked-prefill continuation is serviced
through that same `_schedule_running` loop, *after* the decode requests
already in it. So a decode request's token is never dropped from a step's
schedule by chunk size -- what varies is how long that step itself takes
in wall-clock time. An unchunked (single-shot) long prefill makes one
step's forward pass huge, so every decode request's next token arrives
however long that one giant step took; chunking spreads the same total
prefill work over many small steps, each close to a normal decode step's
duration.

```bash
python3 -m benchmarks.chunked_prefill.measure_itl
```

Writes `itl_summary.csv` (chunk_size, num_repeats, baseline/disrupted ITL
mean+stdev, disrupted ITL max mean+stdev, prefill TTFT mean+stdev),
`itl_raw.csv` (every individual gap, every repeat), and
`step_latency.csv` (one row per `engine.step()` call) into this directory.
Same `--repeats`-with-first-repeat-discarded methodology as
`benchmarks/prefix_caching/measure_ttft.py` -- see its README for why.

## Status

Run for real on a Nebius L40S (2026-09-05, default sweep: 8 decode
requests, 16-token prompts, 100 max tokens each, settled for 10 steps
before injecting one 2048-token prefill request):

| chunk_size | baseline_itl_ms | disrupted_itl_ms_max | prefill_ttft_ms |
|-----------:|-----------------:|----------------------:|-----------------:|
| 128        | 20.2 +/- 0.1      | 23.1 +/- 0.0            | 411.3 +/- 0.8     |
| 256        | 20.1 +/- 0.0      | 23.6 +/- 0.6            | 207.7 +/- 0.8     |
| 512        | 20.1 +/- 0.2      | 25.3 +/- 0.6            | 121.9 +/- 2.0     |
| 1024       | 20.1 +/- 0.1      | 29.7 +/- 0.2            | 81.3 +/- 0.4      |
| 2048       | 20.1 +/- 0.1      | 39.6 +/- 0.1            | 62.2 +/- 0.1      |
| 8192       | 20.1 +/- 0.0      | 40.2 +/- 0.0            | 40.2 +/- 0.0      |

A clean, monotonic tradeoff on both axes -- there is no single "ideal"
chunk size, only a real Pareto curve. Smaller chunks keep decode close to
its undisrupted baseline (128: ITL only rises 15%, from 20.2ms to 23.1ms)
at the cost of the long request taking far longer to complete (411ms,
~10x the unchunked case). Larger chunks let the long request finish fast
(8192, effectively unchunked: 40.2ms, matching its own single-step cost
exactly -- `prefill_ttft` and `disrupted_itl_ms_max` converge because both
are now the same one giant step) at the cost of doubling decode's worst
ITL for that one token. Picking a chunk size in production means picking
a point on this curve for your own SLO mix (interactive chat vs. batch
throughput), not finding a number that wins on both axes at once.

Raw data: `itl_summary.csv` / `itl_raw.csv` / `step_latency.csv` in this
directory (`repeat_index` column included -- repeat 0, discarded from the
aggregate above, is still in there).

## Three real correctness bugs, found chasing this benchmark

Getting to trustworthy numbers here surfaced three real bugs in the core
engine -- none caught by the existing test suite, because no existing test
used a prompt long enough to force genuinely multi-step chunked prefill
(every prompt in `model/tests/test_llm_engine.py` was 3-5 tokens against
a 64-token default budget) or ever hit a 100%-cache-hit's zero-new-tokens
edge case for real. All three are fixed now; the full CUDA test suite
(`kernels/tests/ model/tests/ engine/tests/`) runs clean end to end for
the first time this project has actually verified it.

### Bug 1: multi-step chunked prefill sampled garbage mid-prefill

`model/model_runner.py`'s `execute_model()` sampled and appended a token
to `output_token_ids` for *every* scheduled request every step, including
one still mid-prefill -- not only once it was actually done.
`Request.get_num_new_tokens()` (`len(prompt)+len(output_token_ids)-
num_computed_tokens`) assumes `output_token_ids` only grows once a
request is genuinely done prefilling; the extra append inflated that
count by one per already-run mid-prefill chunk, and once a chunk size
didn't evenly divide the prompt length, that inflation pushed the tail
chunk's token slice past the real prompt boundary into those garbage
samples. Confirmed as a real divergence, not a theoretical concern:
`benchmarks/chunked_prefill/verify_multi_chunk_correctness.py`
(prompt_len=10, chunk=3 -- doesn't divide evenly) produced completely
different output than a dense reference before the fix. This same bug
was *also* why the first cut of this benchmark's own `--prefill-max-tokens
1` silently made the long-prefill request stop after just its first
chunk, at every chunk size below 2048 -- invalidating that entire first
sweep without erroring at all. Fixed by only recording a sample once
`not request.is_prefill()`.

### Bug 2 and 3: a 100%-cache-hit request has no row to sample from

Unrelated to chunk size, found while re-verifying Bug 1's fix against the
existing (but, it turned out, never actually run on real CUDA) test
suite: a request admitted with a full prefix-cache match and zero new
tokens scheduled crashed twice over. First,
`model_runner.py`'s `_attention()`: `out_i...[-L:]` with `L=0` is
Python's `[0:]` (the *whole* tensor, not empty), corrupting an assignment
into an empty slice. Fixing that surfaced the deeper bug one level up:
`execute_model()`'s `last_idx = [e - 1 for e in flat_ends]` is undefined
for a request contributing zero rows -- `flat_ends[i]` just equals
wherever the flat batch's cursor was *before* it, so `e - 1` is either -1
or some other request's row entirely. Root cause: `Scheduler.
_schedule_waiting` let a match seed `num_computed_tokens` all the way to
the full prompt length, leaving genuinely nothing for that step to
compute -- and a request with nothing to compute has no hidden state
anywhere to sample from. Fixed at the source: the match seed is capped at
`len(prompt_token_ids) - 1`, so a "full" cache hit still costs exactly
one token of real compute, never zero. See `engine/README.md`'s prefix
caching section and this fix's git history for the full trace.