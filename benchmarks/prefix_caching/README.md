# Prefix caching: TTFT vs. shared-prefix length

Measures the TTFT (time to first token) win from `engine/prefix_cache.py`'s
RadixTrie (see `engine/README.md`'s "Prefix caching" section for how it's
wired into `BlockManager`/`Scheduler`), for the workload it's meant for:
many requests sharing a common system-prompt prefix.

`measure_ttft.py` sweeps a set of shared-prefix lengths (default `0, 128,
256, 512, 1024, 2048` tokens), and at each length runs the same synthetic
workload `--repeats` times (default 5) against a real `LLMEngine` -- once
with `CacheConfig.enable_prefix_caching=False`, once `True`, per repeat --
recording each request's TTFT every time and reporting a mean +/- stdev
per prefix length. See its module docstring for the exact methodology
and caveats (one timestamp per batched `step()`, not per-request
GPU-event precision) and "How much does `--repeats` matter" below for why
this isn't just belt-and-suspenders -- a single measurement here is noisy
enough to get the *direction* of a result wrong, not just its exact size.

```bash
# Run as a module, not a script -- this repo has no pyproject.toml/setup.py
# or conftest.py, so nothing else puts the repo root on sys.path for
# `from engine/model import ...` to resolve (same reason kernels-ci.yml
# runs `python -m pytest`, not bare `pytest` -- see its own comment).

# Full sweep, on a real CUDA GPU (see the "Status" note below)
python3 -m benchmarks.prefix_caching.measure_ttft

# Small smoke test first -- --repeats 1 since this is just checking the
# path works end to end, not measuring anything trustworthy yet
python3 -m benchmarks.prefix_caching.measure_ttft --prefix-lens 0,32 --num-requests 4 --max-tokens 1 --repeats 1
```

Writes `prefix_cache_ttft_summary.csv` (`prefix_len, num_repeats,
cache_off_ttft_ms_mean, cache_off_ttft_ms_stdev, cache_on_ttft_ms_mean,
cache_on_ttft_ms_stdev, reduction_pct_mean, reduction_pct_stdev`),
`prefix_cache_ttft_raw.csv` (every individual measurement from every
repeat), and `prefix_cache_step_latency.csv` (one row per `engine.step()`
call -- duration, tokens scheduled, queue depth -- for diagnosing *why* a
configuration is slow, not just that it is) into this directory. An
untimed `warmup()` pass runs once per prefix length before any repeat is
ever timed (see its docstring) -- `--skip-warmup` turns this off for
faster iteration on the harness itself, at the cost of cold-start-biased
numbers.

## Status

Run for real on a Nebius L40S (2026-09-05, default sweep: `num_requests=32`,
`suffix_len=16`, `max_tokens=1`, `num_gpu_blocks` auto-sized off the
caching-off worst case, `--repeats 5` with repeat 0 discarded -- see
below). Getting a trustworthy number took two real engine bugs, and a
benchmark-methodology chase (five warmup fixes across two different root
causes, an actual profiler, and finally `--repeats` itself) -- final
result, mean +/- stdev over 5 kept repeats:

| prefix_len | cache_off_ttft_ms | cache_on_ttft_ms | reduction_pct |
|-----------:|------------------:|------------------:|--------------:|
| 0          | 68.4 +/- 0.1        | 68.5 +/- 0.1        | -0.2% +/- 0.2%  |
| 128        | 101.0 +/- 0.6       | 101.2 +/- 0.2       | -0.2% +/- 0.6%  |
| 256        | 113.5 +/- 0.1       | 96.3 +/- 0.2        | 15.1% +/- 0.2%  |
| 512        | 153.0 +/- 0.2       | 97.7 +/- 0.1        | 36.1% +/- 0.1%  |
| 1024       | 248.9 +/- 0.8       | 110.9 +/- 0.4       | 55.4% +/- 0.3%  |
| 2048       | 466.2 +/- 2.4       | 150.0 +/- 1.6       | 67.8% +/- 0.5%  |

A real, clean, monotonic trend once the noise is gone: `reduction_pct`
climbs steadily from ~0% at 0/128 tokens shared to 67.8% at 2048.
`prefix_len` 0 and 128 both landing at ~0% (well within a stdev of zero)
is real, not a bug: at `prefix_len=0` there's no shared content at all
for the trie to match (`build_workload` gives every request an
independently-random suffix there), and at `prefix_len=128` (8 blocks)
the fixed per-admission bookkeeping overhead this implementation pays for
every cache hit roughly cancels out the compute it saves -- caching only
becomes clearly worthwhile past a few hundred shared tokens *in this
specific implementation*, not as some universal property of prefix
caching. Raw data: `prefix_cache_ttft_summary.csv` /
`prefix_cache_ttft_raw.csv` / `prefix_cache_step_latency.csv` in this
directory (`repeat_index` column included -- repeat 0, discarded from
the aggregate above, is still in there for anyone who wants to look).
Chart: [Prefix Cache Payoff](https://claude.ai/code/artifact/575c141e-11c9-4800-a322-bc2cb9786963)
(both figures above, plus the full table, as an interactive page --
same pattern as `../profiling/roofline/README.md`'s).

### Bug 1: ref-count double-release (real correctness bug)

First real run: 32 requests admitted in the same step, all sharing a
prefix and all missing the (then-empty) cache, independently computed and
registered the same blocks; `RadixTrie.release()` raised `ValueError: ...
ref_count already 0` when they all later freed. `engine/prefix_cache.py`'s
`insert()` only bumped `ref_count` for a block it *created*, leaving a
second request that found the same block already there uncounted --
worse than the crash itself, the same gap meant a shared block's
`ref_count` could hit 0 (eligible for eviction) as soon as just the
*first* of its N sharing requests finished, while N-1 others still
depended on it. Fixed (see git history) by having `insert()` bump
`ref_count` for every node it returns unless the caller already owns it.

### Bug 2: O(context_length²) attention cost slips past the token budget

Re-run clean after Bug 1, but `cache_on` got progressively *worse* than
`cache_off` from prefix_len=512 onward (down to -616% at one point).
Real cause: `model/model_runner.py`'s `_attention()` pads every
non-fresh-prefill call up to the full context length and computes
O(context_length²) attention regardless of how few tokens are new (a
documented trade-off for reusing the flash-attention kernel unmodified).
A prefix-cache hit makes a request's *new*-token count tiny while its
real context length stays huge, so `SchedulerConfig.max_num_batched_tokens`
(which only charges by new tokens) let many such requests pile into the
same step -- measured: ~30 requests at once, each still paying full
O(2064²) attention, a 1.9-2.0 *second* step vs. ~55ms for every other
step in the same run. Fixed by
`SchedulerConfig.max_cache_hit_context_tokens`, a second per-step budget
charged by matched-context length instead of new-token count (see its
docstring and `Scheduler._schedule_waiting`).

### The warmup chase: three wrong guesses, then an actual profile

With both bugs fixed, `prefix_len` 1024/2048 flipped strongly positive,
but 128/256/512 barely moved -- still a ~1.35 second spike on their first
cache-hit continuation step. Three consecutive attempts to fix
`measure_ttft.py`'s `warmup()` (covering the cache-on shape at all;
forcing a genuine registered match via a donor-then-followers split;
fixing a crash that split caused) all failed to move these three lengths,
because all three were guesses about *what* needed warming rather than
verified measurements.

What actually found it: `benchmarks/prefix_caching/profile_spike.py`
(committed, one-off, not part of the regular suite) wraps the exact slow
scenario in `torch.profiler`. That showed ~99% of the slow step's GPU
time in our own `matmul` kernel and tensor zero-initialization -- real,
recurring compute, not a one-time cost at all. `kernels/tiled_matmul.py`'s
`_matmul_kernel` is `@triton.autotune(key=["M", "N", "K"])`, and `M` (the
flat-batch row count every `Linear` in `execute_model()` runs over) is
however many followers' new tokens get batched into one cache-hit
continuation step -- which varies a lot by `prefix_len`
(`floor(max_cache_hit_context_tokens / matched_tokens)`: 32 followers at
prefix_len=128, 2 at prefix_len=2048). Autotune benchmarks every
candidate config for a **new** `M` live, inside the timed call -- a real
multi-config timing sweep, not a fixed tax -- and warmup's small fixed toy
`num_requests` (4) never reached the `M` values 128/256/512's larger
follower groups need, so those `M` values were never autotuned until the
real, timed run hit them.

Fixed by having `warmup()`'s cache-hit follower batch use the real
sweep's own `--num-requests` (not a small toy count), so its own natural
per-step admission grouping lands on the exact same `M` per `prefix_len`.
Verified cheaply with `profile_spike.py --warmup` before committing to
another full sweep: the same prefix_len=256 scenario dropped from 1.836s
to 9.6ms CUDA time.

### Aside, flagged but not fixed

`profile_spike.py`'s early iterations also crashed on a **real, separate**
latent bug: a request admitted with a fully-cached prefix and zero new
tokens this step gets pushed through `_attention`/sampling with an empty
flat-batch range -- `out_i...[-L:]` with `L=0` is Python's `[0:]` (the
*whole* tensor), not empty, corrupting the assignment. Worked around in
`warmup()`'s own workload construction (never submit an exact-duplicate
prompt), not fixed in `model/model_runner.py` itself -- real workloads
essentially never produce byte-identical prompts, so it wasn't otherwise
in scope here, but it's a real crash waiting for whoever hits it for
real (e.g. a retried or literally-duplicated request under caching).

## How much does `--repeats` matter

Every result above came from a **single** measurement per prefix length
-- no repeats, no averaging. That's how the 512+ regression above got
found in the first place, but it also means those exact numbers carry
real, unquantified GPU/system timing noise: a `prefix_len=256` run that
happened to land on an unusually fast or slow `cache_off` sample would
report a wildly different `reduction_pct` than a typical one, with no way
to tell from a single number whether that happened.

`--repeats` (now built into `main()`, default 5 -- see `REPEATS`'s own
comment for the 3-vs-5-vs-10 tradeoff) runs the same workload multiple
times per prefix length and reports mean +/- stdev instead of one sample,
turning "did this look better or worse" into "how much did it actually
vary, and is that big relative to the mean."

It immediately earned its keep: the first real `--repeats 5` run showed
`prefix_len=128`'s reduction as `15.9% +/- 35.8%` -- nothing like the
clean 79.7% the earlier single-sample run reported. Cause, found from
`repeat_index`-level data in the raw/step CSVs: `warmup()`'s cache-*off*
pass still used a small toy `num_requests` (4), never having fixed that
side the way the cache-on side already was -- repeat 0 alone paid ~936ms
for a never-before-seen `M=4096` and ~1894ms for `M=432`, both ~12-100ms
in every later repeat. Same root cause as the cache-on fix earlier in
this doc, just never applied to cache-off; fixed by giving `warmup()` one
`num_requests` (defaulting to the real sweep's own) used for both passes
instead of a small separate toy count for cache-off.

Re-verifying that fix (a cheap `--repeats 2` check, not a full sweep)
found a **fourth**, structurally deeper mismatch: `warmup()`'s cache-on
pass runs a donor fully to completion, then submits *all* followers as
one batch that immediately gets a full match -- but the real sweep
submits everyone at once with an empty cache, so only the first ~28
requests get genuine misses (filling most of step 0's token budget) and
just the few stragglers left over see a match in step 1, at a *much*
smaller `M` than warmup's all-32-followers-matched-at-once batch ever
produces. Four rounds of chasing warmup() to predict every shape a given
prefix_len/repeat combination will hit is enough to call it a moving
target rather than a bug with one more fix.

Instead of a fifth warmup attempt: `main()` now runs one extra,
unaveraged repeat 0 and discards it by default (`--keep-first-repeat` to
include it) -- since repeat 0 is consistently where this class of
residual cold-start cost lands, treating it as one more (discarded)
warmup pass sidesteps the problem without needing warmup() to predict
every shape in advance.

That worked. The verification run (`--repeats 5`, discard-first-repeat
on) is the "Status" table at the top of this doc -- every stdev landed
under 2.5ms (under 0.6 percentage points on `reduction_pct`), and it's
the run that revealed `prefix_len=128` has ~0% real reduction, not the
79.7% a single noisy sample had reported. Net result of the whole
`--repeats` chase: not just tighter error bars on numbers that were
already roughly right, but a materially different, more accurate
picture of where prefix caching actually starts paying off in this
implementation.