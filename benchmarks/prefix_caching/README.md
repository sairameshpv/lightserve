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
# Full sweep (run on a real CUDA GPU -- see the "Status" note below)
python3 benchmarks/prefix_caching/measure_ttft.py

# Small smoke test first
python3 benchmarks/prefix_caching/measure_ttft.py --prefix-lens 0,32 --num-requests 4 --max-tokens 1
```

Writes `prefix_cache_ttft_summary.csv` (`prefix_len, cache_off_ttft_ms,
cache_on_ttft_ms, reduction_pct`) and `prefix_cache_ttft_raw.csv` (every
individual measurement) into this directory.

## Status

Not yet run for real -- this script's engine-driving path needs a real
CUDA GPU (see its module docstring), which this was authored without
access to. `build_workload`/`summarize_results` (the torch-free helpers) are
covered by `benchmarks/tests/test_measure_ttft.py` and run cleanly
anywhere; `main()`'s actual `LLMEngine` sweep is reviewed but unverified.

Next step: run the smoke test above on a Nebius L40S (matching
`benchmarks/README.md`'s own precedent for where this repo's real-GPU
numbers come from), then the full sweep, then turn the summary CSV into a
"TTFT reduction % vs. prefix length" chart (an interactive Artifact, same
as `../profiling/roofline/README.md`'s) and link it here.