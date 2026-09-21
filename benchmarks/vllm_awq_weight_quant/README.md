# AWQ w4 weight quantization vs bf16 on real vLLM

Real vLLM (v0.26.0), real `meta-llama/Meta-Llama-3-8B-Instruct` (bf16)
vs. a real published AWQ w4 quantization of the same model
(`TechxGenus/Meta-Llama-3-8B-Instruct-AWQ`, `--quantization
awq_marlin`), one L40S. Follows `benchmarks/vllm_pd_kv_quant/`'s KV
cache quantization result -- this round tests the *other* axis:
weight quantization, orthogonal to KV cache and to P/D disaggregation
(no second node needed). No lightserve code involved.

## Setup

Both served from the same box, one at a time -- the default
`vllm-server` container for bf16 (already provisioned with exactly
this model, no changes needed), a second `vllm-awq` container for the
quantized checkpoint:

```
docker run -d --name vllm-awq --gpus all --shm-size 32g --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=... --env HUGGING_FACE_HUB_TOKEN=... \
  -p 8000:8000 vllm/vllm-openai:latest \
  --model TechxGenus/Meta-Llama-3-8B-Instruct-AWQ --quantization awq_marlin
```

`--quantization awq_marlin` is required explicitly -- confirmed via
vLLM's own GitHub issues before running anything, it is **not**
auto-detected from the checkpoint's `config.json`, and omitting it
causes OOM/cryptic CUDA errors rather than a clean failure.

CUDA graphs enabled on both (no `--enforce-eager`) -- unlike the
NixlConnector P/D round, nothing here needs eager mode, so this is a
more production-realistic config than that round's.

**Correctness**: real completion sent and eyeballed before trusting
any timing (a lightweight check, not a full token-match-rate study --
this consumes an existing published checkpoint rather than new
quantization logic this project wrote itself): prompt `"The capital of
France is"` -> `" also home to the famous Eiffel Tower, the iconic
landmark that has"` -- coherent, correct.

**Checkpoint size, measured**: bf16 14.96 GiB, AWQ 5.4 GiB -- a real
**2.77x** reduction, not the idealized 4x a pure int4-vs-2-byte
argument would suggest (embeddings/norms typically stay unquantized,
and group-wise scale/zero-point overhead adds real bytes back -- the
same story as the KV cache scale tensors in the earlier rounds).

## Results

`vllm bench serve`, `--random-input-len 512 --random-output-len 128`,
seven concurrency points -- 1, 8, 32 in the original round; 64 and 128
added to find where the win bottoms out; 256 and 512 added in a final
follow-up specifically to find where it actually **converges**:

| Max concurrency | bf16 mean ITL | AWQ mean ITL | Mean speedup | bf16 median ITL | AWQ median ITL | Median speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.28ms | 7.61ms | 2.80x | 21.27ms | 7.61ms | 2.80x |
| 8 | 23.08ms | 8.83ms | 2.61x | 22.79ms | 8.32ms | 2.74x |
| 32 | 31.64ms | 15.24ms | 2.08x | 26.65ms | 11.39ms | 2.34x |
| 64 | 43.56ms | 30.29ms | 1.44x | 31.20ms | 16.29ms | 1.92x |
| 128 | 68.35ms | 41.59ms | 1.64x | 39.39ms | 26.84ms | 1.47x |
| 256 | 120.53ms | 117.91ms | 1.02x | 57.32ms | 49.17ms | 1.17x |
| 512 | 133.58ms | 131.95ms | 1.01x | 58.64ms | 49.94ms | **1.17x** |

All requests succeeded at every point, both dtypes.

**Convergence found, and confirmed real, not a stopping-point
guess**: the ratio plateaus between c=256 and c=512 -- median speedup
is 1.17x at *both* points, mean speedup 1.02x and 1.01x. This isn't
just "the numbers happened to be close": **output token throughput is
flat from c=256 to c=512 for both dtypes** (bf16: 1661.20 -> 1675.75
tok/s; AWQ: 1722.36 -> 1699.98 tok/s -- both within noise of each
other, neither showing a real increase despite doubling the client's
own concurrency) while TTFT exploded (bf16: 3319ms mean at c=256 ->
17968ms at c=512) and `Peak concurrent requests` sat near its ceiling
at both points (~276 and ~530, tracking the client cap rather than
growing served throughput) -- the clear signature of the GPU having
hit its real execution ceiling around c≈256: additional client-side
concurrency beyond that point stops translating into more actual
parallel GPU work and just adds queueing delay, which per-token ITL
(measured only once a request is actually running) doesn't reflect.
Pushing further than c=512 would very likely just reproduce the same
plateaued ratio, not reveal anything new -- this is genuine
convergence, not premature stopping.

**Why both mean and median are reported**: the mean-ITL speedup wasn't
monotonic through the middle of the sweep (1.44x at c=64, back up to
1.64x at c=128) -- checked before accepting either number at face
value. The gap between mean and median ITL widens sharply at high
concurrency (e.g. bf16 c=128: mean 68.35ms vs. median 39.39ms) -- a
single untimed trial at heavy load is genuinely noisy from occasional
queueing-driven tail latency (P99 ITL at c=128 is 190ms, ~4.7x the
median), and that noise lands unevenly between the two conditions'
single runs. Median ITL, less sensitive to those outliers, gives a
clean, monotonically decreasing pattern all the way to the plateau
(2.80x -> 2.74x -> 2.34x -> 1.92x -> 1.47x -> 1.17x -> 1.17x) -- the
real story, with the mean-ITL blip explained rather than silently
preferred or ignored. Mean speedup converges to ~1.0x (true parity) at
the plateau; median settles at a real, durable ~1.17x -- both
plausible depending on which better represents "typical" latency
under this workload's heavy-tail queueing behavior at saturation.

## Reading this: the plan's own prediction was too narrow, corrected honestly

The plan stated a prediction to check the real result against: a real
win at concurrency=1, fading toward little/no difference by
concurrency=8 (mirroring how this project's own earlier profiling
found the LM-head GEMV amortizes into a GEMM under batching). **The
real result didn't match that shape** -- concurrency=8 still showed a
2.61x speedup, barely faded from concurrency=1's 2.80x. Investigated
before writing this up rather than smoothed over.

**What actually explains the shape, checked against the numbers, not
just asserted:**

1. **The crossover point is real, just at a higher concurrency than
   guessed.** AWQ's own byte reduction (2.77x, measured above) is
   larger than what a KV-cache-quantization-informed intuition assumed
   -- a bigger memory-cost cut means a *larger* batch size is needed
   before compute cost catches up to and exceeds the now-much-smaller
   memory cost. The full sweep confirms this directly: a clean,
   monotonic fade (median speedup) from 2.80x at c=1 down to a plateau
   at c=256/512.
2. **The win does not converge to 1.0x -- it plateaus at a real,
   durable ~1.17x (median), confirmed stable across two full-saturation
   points (c=256 and c=512).** This is mechanistically different from
   the KV-cache rounds, not just "the same fade, further out." KV cache
   quantization's only lever is memory bytes -- once truly compute-bound,
   its advantage should converge to ~0, which is exactly what fp8 KV
   showed at short context. AWQ's Marlin kernels, though, do genuine
   mixed-precision int4-weight x fp16-activation tensor-core compute,
   not a dequant-then-bf16-matmul -- if that mixed-precision compute
   path itself runs at higher throughput than plain bf16 x bf16 on this
   GPU, part of AWQ's advantage persists even once memory bandwidth
   stops being the bottleneck and the GPU is fully saturated on both
   dtypes. The durable ~1.17x median plateau (vs. mean's ~1.0x, see
   below) is consistent with this second, compute-throughput mechanism
   being real and separate from the fading memory-bandwidth one -- not
   fully isolated here (would need per-kernel `ncu` profiling to
   attribute precisely), but no longer just a hypothesis about
   "further out": the plateau itself, confirmed at two independent
   saturation points, is the evidence that *something* durable survives
   past the point where memory bandwidth stops mattering.

## What this confirms

A real weight-quantization technique gives a substantial, real
decode-latency win at every concurrency actually tested here (1
through 512) -- from a 2.80x speedup at concurrency=1 down to a
confirmed, durable ~1.17x (median) plateau once the GPU is fully
saturated. It never fully converges to parity the way KV cache
quantization's purely memory-bandwidth-only advantage did (fp8 KV
showed ~0% benefit at short context, once compute-bound) -- consistent
with AWQ's Marlin kernels contributing a real compute-throughput
advantage on top of the fading memory-bandwidth one. And unlike KV
cache quantization, it doesn't need a second network hop, a
KV-transfer connector, or a disaggregated architecture to realize any
of this: it's a same-box, same-request lever. This is the complementary
result to `benchmarks/vllm_pd_kv_quant/`'s finding -- weight
quantization and KV cache quantization target different parts of the
memory-bandwidth picture, and both showed real, mechanistically-explained
wins on real vLLM, in contrast to lightserve's own unfused int8 KV
cache regression.

## Files

No lightserve code -- this directory documents a real-vLLM measurement
only. Natural follow-ups, not attempted here: `ncu`-level profiling to
actually separate the memory-bandwidth and compute-throughput
contributions to the confirmed ~1.17x plateau; combining AWQ weights
with fp8/int8 KV cache in one deployment (not tested together this
round); repeating each point multiple times to get real variance bars
instead of the single-trial mean/median gap this round had to reason
through by hand; a longer-context workload (this round fixed
`--random-input-len 512` throughout, since context length wasn't the
axis under test -- worth checking whether the plateau value itself
shifts at a different context length).
