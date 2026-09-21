# bf16 vs. fp8/int8 KV cache on real vLLM P/D disaggregation (1+1)

Real vLLM (v0.26.0), real 2-GPU P/D disaggregation via `NixlConnector`
(1 prefill instance + 1 decode instance, the "two-pool" architecture
explained-but-not-built earlier in this project, now tested for real at
1+1 scale), real Llama-3-8B-Instruct checkpoint. Follows directly from
`benchmarks/kv_int8_compression/`'s finding that lightserve's own
unfused int8 KV implementation made decode *slower* -- this measures
whether a real, fused, production implementation behaves differently.

No lightserve code involved -- this is real-vLLM measurement only.

## Setup

Two L40S nodes, no InfiniBand (confirmed absent for `gpu-l40s-a` in an
earlier session). `NixlConnector` over UCX, which negotiated down to
TCP -- worked without any RDMA hardware.

```
# prefill (node 0)
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 VLLM_NIXL_SIDE_CHANNEL_HOST=<node0-internal-ip> \
docker run ... vllm/vllm-openai:latest --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8100 --host 0.0.0.0 --enforce-eager --kv-cache-dtype <dtype> \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# decode (node 1) -- same shape, kv_role":"kv_consumer", its own side-channel host/port

# routing proxy (node 0), vLLM's own tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
python3 toy_proxy_server.py --port 8192 \
  --prefiller-hosts <node0-internal-ip> --prefiller-ports 8100 \
  --decoder-hosts <node1-internal-ip> --decoder-ports 8200
```

**A real config gap found and fixed, not in any doc's own example**:
vLLM's official NixlConnector guide only shows a *same-machine* example,
where `kv_ip`'s default (`127.0.0.1`) is correct by construction. Across
two real nodes it isn't -- the decode instance tried to reach the
prefill instance's NIXL handshake socket at *its own* localhost, not
prefill's real address, failing with `zmq.error.Again: Resource
temporarily unavailable`. Fix: set `VLLM_NIXL_SIDE_CHANNEL_HOST` to each
instance's own real (internal VPC) IP explicitly -- confirmed via a
small-model (`Qwen/Qwen3-0.6B`) smoke test before ever touching the real
checkpoint, catching this before it could cost real-checkpoint GPU time
chasing the wrong hypothesis.

Also found: the real checkpoint's default `max_model_len=8192` means
`--random-input-len 8000 --random-output-len 200` (8200 total) fails
every request outright (400s from the decode instance) -- adjusted the
long-context point to 7500+200=7700, still comfortably past vLLM's own
stated ~7k-token fp8-benefit threshold.

Measured via `vllm bench serve` (native TTFT/TPOT/ITL reporting, no
custom client needed) pointed at the proxy's port -- the single
client-facing entrypoint for the whole 1+1 pipeline.

## Results

**Short context** (`--random-input-len 512 --random-output-len 128
--max-concurrency 8 --num-prompts 40`):

| KV dtype | Mean ITL | Mean TTFT | Median TTFT | P99 TTFT |
|---|---|---|---|---|
| bf16 (`auto`) | 23.21ms | 921ms | 982ms | 1497ms |
| fp8 | 23.94ms | 709ms | 757ms | 1043ms |
| int8_per_token_head | 22.83ms | 1349ms | 832ms | 2881ms |

**Long context** (`--random-input-len 7500 --random-output-len 200
--max-concurrency 8 --num-prompts 20`):

| KV dtype | Mean ITL | Mean TTFT | Median TTFT | P99 TTFT |
|---|---|---|---|---|
| bf16 (`auto`) | 31.86ms | 15889ms | 17434ms | 17840ms |
| fp8 | 27.80ms | 12226ms | 13519ms | 13668ms |
| int8_per_token_head | 26.26ms | 5701ms | 4690ms | 11986ms |

All 40 (short) / 20 (long) requests succeeded at every dtype -- zero
failures.

## Reading this

**fp8 matches vLLM's own stated pattern exactly, and this is the
headline, well-supported result**: essentially flat ITL at short
context (23.94ms vs bf16's 23.21ms -- within noise, no real win, no
real loss either) and a genuine ~13% ITL improvement at long context
(27.80ms vs 31.86ms), plus a real TTFT drop at both scales. This is
precisely what vLLM's own April 2026 blog predicted (little/no benefit
under ~7k tokens, real benefit past it) -- confirming, on real
production hardware and a real fused kernel, that the memory-bandwidth
theory behind KV quantization is real. It also directly supports the
explanation given for lightserve's own regression: a properly fused
implementation realizes the bandwidth win; lightserve's unfused,
plain-PyTorch v1 didn't, because the added op-dispatch overhead
outweighed it at the scale tested there.

**int8_per_token_head's numbers are not treated as a confirmed result**,
despite looking even better than fp8's on paper (ITL 22.83/26.26ms,
better than fp8 at both scales). Two reasons for caution, stated
plainly rather than smoothed over:
1. It's mechanistically unclear why a *less* mature quantization path
   (GitHub issue #33480 requesting general INT8 support was still open
   at research time) would outperform fp8's more established kernel --
   both are 1 byte/element, so there's no obvious storage-side reason
   for the gap.
2. The short-context run itself shows an internal inconsistency worth
   naming: mean TTFT (1349ms) is markedly higher than *median* TTFT
   (832ms) with a P99 of 2881ms -- a real outlier skew not present in
   the bf16 or fp8 runs at the same scale, suggesting occasional
   request-level cost spikes specific to this dtype path that weren't
   investigated further (would need per-request timing data, not
   collected here, or a correctness pass on the actual generated
   tokens -- neither was in scope for this round, which was a speed
   comparison, not a full correctness re-verification).

Read as: fp8 vs. bf16 is a real, trustworthy comparison; int8's numbers
are reported honestly but not vouched for as a "real win" the way
fp8's are.

## What this confirms about the lightserve finding

This round's real, production, fused fp8 result gives a positive
control for last round's negative result: the memory-bandwidth theory
behind KV cache quantization is real (confirmed here, at ~13% ITL
improvement past ~7k tokens, on real vLLM) -- lightserve's own int8 KV
regression was specifically about *that implementation* (unfused,
plain-PyTorch ops layered onto an already-serial per-request-per-layer
loop), not evidence against the technique itself.

## Files

No lightserve code -- this directory documents a real-vLLM measurement
only. The NixlConnector config, the `VLLM_NIXL_SIDE_CHANNEL_HOST` fix,
and the `toy_proxy_server.py` invocation above are the reusable
artifacts if this is revisited (e.g. at 2+2 pool scale, or with a
custom settle-then-inject workload matching lightserve's own P/D
benchmark shape more literally -- both explicitly out of scope this
round).
