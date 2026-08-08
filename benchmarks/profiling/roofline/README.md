# Roofline analysis: attention & FFN on L40S

Analytical (no GPU run required) arithmetic-intensity model for Llama-3-8B's
core ops across sequence lengths, plotted against the L40S's bf16 roofline.
Complements the empirical kernel-trace findings in `../report.md` — that
report found the LM-head GEMV is memory-bound at batch=1; this generalizes
the same style of analysis to attention and FFN, and to how it changes with
sequence length.

**Interactive chart:** https://claude.ai/code/artifact/d4242552-5e8e-4c47-b848-d6727ef1a45b

## Hardware / model constants used

| | Value | Source |
|---|---|---|
| L40S peak BF16 (dense) | 362 TFLOPS | NVIDIA official specs |
| L40S memory bandwidth | 864 GB/s | NVIDIA official specs |
| Ridge point | 419 FLOPs/byte | 362e12 / 864e9 |
| Llama-3-8B hidden size | 4096 | HF `config.json` |
| Llama-3-8B FFN intermediate size | 14336 | HF `config.json` |
| Attention heads / KV heads (GQA) | 32 / 8 | HF `config.json` |
| head_dim | 128 | 4096 / 32 |
| Precision | bf16 (2 bytes/elem) | matches the model actually served |

## Method

For each op, per-matmul FLOPs and bytes are counted directly from its shape:
for `[S,K]@[K,N]`, `FLOPs = 2*S*K*N`, `bytes = 2*(S*K + K*N + S*N)` (read
input, read weight, write output, bf16). Attention's QK^T + softmax@V is
modeled flash-attention style — Q/K/V/O are read/written once each, the
S×S (or S×L) score matrix is **never** materialized to HBM, matching what
the profiled server's `flash_fwd_splitkv_kernel` actually does (a single
fused kernel, not separate HBM round-trips per step).

Two regimes:
- **Prefill** — S tokens processed together in one batched forward pass
  (S = 1, 8, 32, 128, 512, 2048, 8192).
- **Decode** — one new query token attending to a KV-cache of length L
  (same L values), i.e. the regime the earlier empirical profiling actually
  ran (batch=1, autoregressive).

Each point is plotted at its **attainable performance**:
`min(peak_compute, AI * peak_bandwidth)` — by construction this always sits
exactly on the roofline (the diagonal memory-bound ramp below the ridge
point, or the flat compute-bound ceiling above it).

## Findings

1. **Decode-phase attention never leaves the memory-bound region.** Its
   arithmetic intensity saturates at **~4 FLOPs/byte** regardless of KV-cache
   length — both its FLOPs and its bytes scale linearly with cache length
   (L), so the ratio converges instead of growing. This is >100x below the
   L40S's ridge point of 419 FLOPs/byte, and it stays there forever, no
   matter how long the conversation gets.

2. **Every other op is memory-bound at short sequence lengths but crosses
   into compute-bound territory as the batch grows** — FFN crosses first
   (~512 tokens), then attention-score prefill and QKVO projections shortly
   after. This matches the standard result that MLP/FFN layers are the
   easiest part of a transformer to make compute-bound, since they're
   "pure" GEMMs with no per-token KV-cache dependency.

3. **This directly explains the earlier empirical finding** (`../report.md`):
   at batch=1 (a single request, no concurrent load), *everything* —
   projections and attention alike — sits in the S=1/L=small region of this
   chart, deep in memory-bound territory. That's exactly why the LM-head
   GEMV (itself just another `[1,K]@[K,N]` projection) dominated both
   prefill and decode profiles at ~1.45ms/step: at batch=1, none of these
   ops get anywhere near the compute roof. The baseline benchmarks that
   showed good aggregate throughput ran at concurrency 50 — batching pushes
   the *projection* ops (FFN, QKVO) rightward along this same chart into
   compute-bound territory, though **decode-phase attention's KV-cache
   read is per-request and doesn't batch away the same way**, so it stays
   memory-bound even under concurrent load. That's the standard reason
   attention (not FFN) tends to be the harder part of LLM serving to keep
   compute-bound at scale.

## Empirical validation (real L40S)

The analytical ceiling is only ever a model. `measure_roofline.py` times the
identical shapes for real via CUDA-event timing, run inside the `vllm-openai`
container on `vllm-node-0` (the same Terraform-provisioned L40S from
`../report.md`), while the vLLM server sat idle:

```
nebius compute v1 instance start --id <instance-id>
scp measure_roofline.py ubuntu@<host>:/tmp/
ssh ubuntu@<host> sudo docker cp /tmp/measure_roofline.py vllm-server:/measure_roofline.py
ssh ubuntu@<host> sudo docker exec vllm-server python3 /measure_roofline.py > roofline_measured.json
nebius compute v1 instance stop --id <instance-id>   # stop promptly, GPU billing is hourly
```

`compare_measured.py` merges the result against the analytical model
(`roofline_comparison.json`). Three things stand out:

1. **The ceiling itself is optimistic.** Peak achieved was **350.8 TFLOPS**
   (attention-score prefill, S=2048) — 97% of the 362 TFLOPS spec, the closest
   anything gets. But FFN, despite being "compute-bound" by arithmetic
   intensity from S=512 onward, only reaches **42–52% of its theoretical
   ceiling**, and that efficiency *keeps dropping* as S grows (81% at S=128 →
   42% at S=8192) — crossing the ridge point doesn't guarantee you get
   anywhere near peak FLOPS.
2. **Tiny shapes are latency-bound, a regime the roofline model doesn't
   capture at all.** At S=1–32, attention/QKVO ops hit only 0.1–15% of their
   (already tiny) memory-bound ceiling — fixed per-kernel launch overhead
   dominates when there's too little work to hide it behind.
3. **Decode attention at S=8192 measures 1156 GB/s** — above the L40S's 864
   GB/s HBM spec — because its ~34MB of K/V cache fits inside the L40S's L2
   cache and gets served from there across the timed loop's repeated calls.
   Doesn't change the qualitative finding (still deep in the memory-bound
   region relative to the ridge point), but it's a reminder that "bytes" in
   the model means HBM traffic, not what a warm cache actually delivers.

Bottom line: the classification (which ops are memory- vs compute-bound, and
where the crossover happens) holds up empirically — but treat the analytical
model's *absolute* attainable-TFLOPS numbers as an upper bound, not a
prediction, especially for large GEMMs (~50% real efficiency) and tiny/latency-
bound shapes (<15% real efficiency).

## Files

- `compute_roofline.py` — the analytical model, run with no arguments.
- `roofline_data.csv` / `roofline_data.json` — full computed analytical data
  (FLOPs, bytes, arithmetic intensity, attainable TFLOPS, classification) for
  every op × sequence length.
- `measure_roofline.py` — empirical CUDA-event timing of the same shapes;
  meant to run inside the vllm-openai container on a real L40S.
- `roofline_measured.json` — raw measured results (elapsed_ms, achieved
  TFLOPS/GB-s) from the run described above.
- `compare_measured.py` — merges measured vs. analytical into
  `roofline_comparison.json` and prints a side-by-side efficiency table.
- `roofline.html` — source for the published interactive chart (now overlays
  measured points as open rings against the analytical filled-dot ceiling).