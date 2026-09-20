# int8 KV cache compression

Adds an opt-in int8 storage mode to `model/kv_cache.py`'s `PagedKVCache`
(`engine/config.py`'s `CacheConfig.int8_kv`, default off) -- symmetric,
per-token-per-KV-head quantization, computed fresh on every `write()`
call, dequantized back to the model's own dtype on every `read()`. The
default path is byte-for-byte unchanged; every existing caller
(`model/model_runner.py`'s attention, P/D disaggregation's
`export_request_kv`/`import_request_kv`) needed zero changes, since
storage dtype is entirely internal to `PagedKVCache`.

Two real, separate questions this benchmark answers: **is it correct**
(`verify_kv_int8_correctness.py`) and **is it faster** (reusing
`benchmarks/pd_disaggregation/measure_pd_real_speedup.py`'s existing
2-GPU rig via its own `--int8-kv` flag, for a direct comparison against
the bf16 numbers already recorded there).

```bash
python3 -m benchmarks.kv_int8_compression.verify_kv_int8_correctness
```

(`sudo`, same two real-checkpoint gotchas as every other script in this
repo -- see `benchmarks/pd_disaggregation/README.md`.)

## Status: correctness

Run for real on a Nebius L40S (2026-09-20), the real 5-prompt set from
`benchmarks/speculative_decoding/prompts_tokenized.jsonl`:

```
[medium-code-0001] match_rate=100.0% (24/24) no divergence
[long-0002]        match_rate=100.0% (24/24) no divergence
[medium-code-0003] match_rate=100.0% (24/24) no divergence
[medium-code-0005] match_rate=100.0% (24/24) no divergence
[medium-code-0006] match_rate=4.2% (1/24) first divergence at token 1

Overall top-1 token match rate: 80.8% (97/120)
```

4 of 5 prompts: zero divergence over 24 tokens each. The one exception
is fully explained, not just reported as-is -- see
`_diagnose_divergence.py` (a throwaway diagnostic kept because it
directly explains this result, same reasoning
`benchmarks/chunked_prefill/verify_multi_chunk_correctness.py` uses for
keeping its own diagnostic): the dense model's own logit gap at the
diverging step was already only 0.125 (out of a typical ~15-18 logit
scale) -- a genuinely marginal, near-tie decision by the full-precision
model itself, not created by quantization. int8's small numerical
perturbation collapsed that already-tiny gap to an *exact* tie
(0.0000), at which point argmax's deterministic but arbitrary
tie-breaking (lowest token id) decided the outcome. Once one token
diverges, every later token is computed on a genuinely different
context between the two engines, so the near-zero match rate for the
rest of that one sequence is the expected mechanical consequence of a
single flipped near-tie, not evidence of larger ongoing damage. This is
about as clean a signature of "small, expected-magnitude quantization
noise nudging an already-marginal call" as this kind of test could show
-- a real bug would far more plausibly produce a large, unexplained
logit gap, not a near-perfect tie.

**Memory arithmetic**, computed for the first time in this repo (see
`engine/config.py`'s own module docstring on why this was previously
left as "deployment-time arithmetic" nobody had written yet):

```
n_layers=32 num_kv_heads=8 head_dim=128 block_size=16
bf16:  2,097,152 bytes/block
int8:  1,081,344 bytes/block (1,048,576 storage + 32,768 fp32 scale overhead)
Same GPU memory budget fits 1.94x as many blocks under int8_kv
```

Matches the hand-derived formula exactly (`0.5 + 2/head_dim` for this
shape) -- not quite 2x, because the fp32 scale tensors are real, if
small, overhead.

## Status: speed -- the real finding is a regression, not a speedup

Run on the same real 2-GPU rig as `benchmarks/pd_disaggregation`'s own
benchmark (2026-09-20), same workload, directly comparable to the bf16
numbers already recorded there:

| | bf16 (2026-09-19) | int8 (2026-09-20) |
|---|---|---|
| Monolithic (1 GPU) baseline ITL | 323.3ms | 365.6ms (+13%) |
| Monolithic disrupted ITL (mean/max) | 459.8ms / 2197ms | 502.0ms / 2253ms |
| Disaggregated (2 GPU) baseline ITL | 98.6ms | **468.1ms (+375%)** |
| Disaggregated disrupted ITL (mean/max) | 98.2ms / 98.8ms | 528.6ms / 3249ms |

int8 KV made decode **slower** in both conditions -- the opposite of
what the memory-bandwidth theory predicted. Checked this wasn't a
repeat of the warmup-measurement bug the bf16 run itself had to fix:
`num_baseline_samples` for the disaggregated leg dropped from 76 to 12,
but the arithmetic checks out (~468ms/token over a fixed 2-second
window with 4 concurrent requests predicts roughly 12-16 gaps) -- the
low sample count is *corroborating* the slow rate, not a broken capture
window.

**Explanation, well-supported:** `_quantize`/`_dequantize`
(`model/kv_cache.py`) are deliberately plain, unfused PyTorch ops --
scoped as "v1, correctness-first" in the implementation plan, the same
tier `kernels/flash_attention.py`'s own v1 was before its v2 tuning
pass. `model/model_runner.py`'s `_attention` already calls
`write()`/`read()` once *per request per layer* (a documented,
deliberate scope cut -- see `benchmarks/pd_disaggregation`'s Locust
comparison for the 30-100x-behind-real-engines cost of that same
architecture). Layering several extra small, unfused op dispatches
(`.float()`, `.abs()`, `.amax()`, `.round()`, `.clamp()`, per K and V)
onto an already-serial per-request-per-layer loop multiplies out fast --
32 layers x 4 requests x ~15 extra small ops is roughly 2,000 extra
kernel launches per decode step. This is the exact lesson
`kernels/README.md`'s own fused bias+ReLU kernel already found the hard
way: below some size, unfused wins purely on launch-overhead grounds.
int8_kv's quantize/dequantize path never got a fusion pass.

**Open, not confirmed:** why the disaggregated leg's regression (+375%)
is so much larger than the monolithic leg's (+13%), given both run the
same per-request-per-layer loop over the same shapes. The leading
hypothesis -- the decode role-server's background `_DecodeWorker`
thread plus FastAPI's async event loop creates more concurrent
Python-level activity (GIL contention) than the monolithic benchmark
script's single-threaded loop, and the added quantize/dequantize ops
amplify that -- is plausible but was not confirmed with a profiler
before this stage stopped (would need `nsys`/torch profiler attached to
the live role-server under real load, a real additional GPU-time cost
not spent here).

## What this means

int8 KV cache compression's theoretical payoff (roughly 2x memory
capacity, halved attention bandwidth) is real and confirmed by the
memory arithmetic above -- but *this* implementation doesn't realize
the bandwidth half of that payoff, because the quantize/dequantize
overhead currently costs more than the bandwidth it saves, on this
architecture at this workload's scale. A fused Triton kernel doing
quantize-on-write/dequantize-on-read in one launch (the natural v2,
mirroring every other kernel in this repo's own v1-correctness /
v2-speed arc) is the obvious next step if this is worth pursuing
further -- not attempted here.

## Files

- `verify_kv_int8_correctness.py` -- correctness + memory arithmetic.
- `_diagnose_divergence.py` -- throwaway diagnostic explaining the one
  real divergence above; kept because it's load-bearing for that
  result, not because it's a reusable tool.
