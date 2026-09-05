# Prefix caching: TTFT vs. shared-prefix length

Measures the TTFT (time to first token) win from `engine/prefix_cache.py`'s
RadixTrie (see `engine/README.md`'s "Prefix caching" section for how it's
wired into `BlockManager`/`Scheduler`), for the workload it's meant for:
many requests sharing a common system-prompt prefix.

`measure_ttft.py` sweeps a set of shared-prefix lengths (default `0, 128,
256, 512, 1024, 2048` tokens), and at each length runs the same synthetic
workload twice against a real `LLMEngine` -- once with
`CacheConfig.enable_prefix_caching=False`, once `True` -- recording each
request's TTFT both times. See its module docstring for the exact
methodology and caveats (one timestamp per batched `step()`, not
per-request GPU-event precision).

```bash
# Run as a module, not a script -- this repo has no pyproject.toml/setup.py
# or conftest.py, so nothing else puts the repo root on sys.path for
# `from engine/model import ...` to resolve (same reason kernels-ci.yml
# runs `python -m pytest`, not bare `pytest` -- see its own comment).

# Full sweep, on a real CUDA GPU (see the "Status" note below)
python3 -m benchmarks.prefix_caching.measure_ttft

# Small smoke test first
python3 -m benchmarks.prefix_caching.measure_ttft --prefix-lens 0,32 --num-requests 4 --max-tokens 1
```

Writes `prefix_cache_ttft_summary.csv` (`prefix_len, cache_off_ttft_ms,
cache_on_ttft_ms, reduction_pct`) and `prefix_cache_ttft_raw.csv` (every
individual measurement) into this directory.

## Status

Run for real on a Nebius L40S (2026-09-04, default sweep: `num_requests=32`,
`suffix_len=16`, `max_tokens=1`, `num_gpu_blocks` auto-sized off the
caching-off worst case). First real run hit an actual bug -- 32 requests
admitted in the same step, all sharing the same prefix and all missing the
(empty) cache, independently computed and registered the same blocks;
`RadixTrie.release()` raised `ValueError: ... ref_count already 0` when
they all later freed. Fixed in `engine/prefix_cache.py`'s `insert()` (see
its git history) -- it only bumped `ref_count` for a block it *created*,
leaving a second request that found the same block already there
uncounted. Worse than the crash itself: the same gap meant a shared
block's `ref_count` could hit 0 -- becoming eligible for eviction -- as
soon as just the *first* of its N sharing requests finished, while N-1
others were still depending on it. Re-run clean after the fix:

| prefix_len | cache_off_ttft_ms | cache_on_ttft_ms | reduction_pct |
|-----------:|------------------:|------------------:|--------------:|
| 0          | 3813.5             | 68.9               | 98.2%          |
| 128        | 3128.0             | 226.5              | 92.8%          |
| 256        | 951.5              | 767.0              | 19.4%          |
| 512        | 1307.1             | 1511.0             | -15.6%         |
| 1024       | 1418.8             | 1761.2             | -24.1%         |
| 2048       | 1678.6             | 1920.0             | -14.4%         |

Two things stand out, neither yet confirmed root-caused:

- **prefix_len=0/128's huge win is likely inflated by a warmup confound,
  not purely the cache**: `main()` builds `engine_off` before `engine_on`
  in every iteration, and the very first iteration (prefix_len=0) pays
  whatever one-time CUDA context / Triton kernel-autotuning cost exists in
  the process -- landing entirely on that first `cache_off_ttft_ms`
  (3813ms, the highest of any off-run despite prefix_len=0 needing the
  *least* compute of the sweep). A fairer methodology would run one
  untimed warmup step before either engine's first measurement.
- **cache_on gets *worse* than cache_off from prefix_len=512 onward**:
  the opposite of what a working cache should do at longer shared
  prefixes. Leading hypothesis, not yet verified: `--num-gpu-blocks`
  defaults to the caching-*off* worst case (no slack for blocks a
  finished request's cache ref is still holding onto rather than
  returning to the free pool), so `engine_on` likely spends real time in
  `BlockManager._drain_evictions`/`RadixTrie.evict_one_lru` fighting for
  blocks the off run never had to reclaim -- an operational cost of
  prefix caching (it needs headroom beyond the bare compute-need to net
  positive) rather than a correctness bug. Worth confirming by re-running
  with a larger explicit `--num-gpu-blocks` before trusting the crossover
  as real.

Raw data: `prefix_cache_ttft_summary.csv` / `prefix_cache_ttft_raw.csv` in
this directory (384 rows, one per request per prefix_len per cache
on/off).

Next step: re-run with generous `--num-gpu-blocks` headroom and an
untimed warmup step to separate the real caching effect from these two
confounds, then turn the (corrected) summary CSV into a "TTFT reduction %
vs. prefix length" chart (an interactive Artifact, same as
`../profiling/roofline/README.md`'s) and link it here.