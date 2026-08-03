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

## Files

- `compute_roofline.py` — the analytical model, run with no arguments.
- `roofline_data.csv` / `roofline_data.json` — full computed data (FLOPs,
  bytes, arithmetic intensity, attainable TFLOPS, classification) for every
  op × sequence length.
- `roofline.html` — source for the published interactive chart.