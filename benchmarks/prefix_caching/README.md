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
cache_on_ttft_ms, reduction_pct`), `prefix_cache_ttft_raw.csv` (every
individual measurement), and `prefix_cache_step_latency.csv` (one row per
`engine.step()` call -- duration, tokens scheduled, queue depth -- for
diagnosing *why* a configuration is slow, not just that it is) into this
directory. An untimed `warmup()` pass runs once per prefix length before
either engine is ever timed (see its docstring) -- `--skip-warmup` turns
this off for faster iteration on the harness itself, at the cost of
cold-start-biased numbers.

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

Two things stood out, and the table above is now known to be **superseded**
-- see the update below the line:

- **prefix_len=0's 98.2% "win" cannot be real caching**: `build_workload`
  gives every request an independently-random suffix when `prefix_len=0`
  (`shared_prefix` is an empty list), so there is zero shared content
  between any two requests -- the RadixTrie cannot produce a single
  cross-request hit here. Since `main()` builds and times `engine_off`
  before `engine_on` in every iteration, and prefix_len=0 is the very
  first iteration of the whole process, this number is almost entirely a
  cold-start artifact (first-ever CUDA/Triton kernel compile landing on
  `cache_off`), not the cache doing anything.
- **cache_on gets *worse* than cache_off from prefix_len=512 onward.**
  The `--num-gpu-blocks`-eviction-thrashing theory floated here originally
  doesn't survive closer inspection: caching-on should need *fewer*
  physical blocks live at once than caching-off (later-admitted requests
  reuse the shared prefix instead of getting their own copy), not more --
  there shouldn't be eviction pressure. Retracted as the leading
  explanation; the real cause needs the step-by-step data below, not
  another guess.

---

**Update**: `measure_ttft.py` now runs an untimed `warmup()` pass per
prefix length before either engine is ever timed (removes the cold-start
bias above at every length, not just 0), and records
`prefix_cache_step_latency.csv` alongside the existing two CSVs, so the
512+ regression can actually be diagnosed from real per-step data instead
of guessed at. Re-run on the L40S pending -- this section will be replaced
with corrected numbers and a data-backed explanation once that's done.