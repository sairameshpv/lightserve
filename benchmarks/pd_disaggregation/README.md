# P/D (prefill/decode) disaggregation

Proves the *mechanism* for prefill/decode disaggregation -- running a
request's prefill on one `LLMEngine` and its decode on a different one,
handing the prompt's KV cache across instead of recomputing it (see
`model/pd_disaggregation.py`) -- correctly, on this project's real
Llama-3-8B checkpoint. Does **not**, and structurally cannot usefully,
measure a speedup: see "Why no speed benchmark" below.

```bash
python3 -m benchmarks.pd_disaggregation.verify_pd_correctness
```

(`sudo` because the real checkpoint lives under `/root/.cache/huggingface`
-- see `benchmarks/speculative_decoding/verify_speculative_correctness.py`'s
module docstring for why. Also `sudo docker stop vllm-server` first: it
auto-starts on boot and holds most of the L40S's GPU memory for its own
serving otherwise -- same two gotchas as every other real-checkpoint
script in this repo.)

Reuses `benchmarks/speculative_decoding/prompts_tokenized.jsonl` rather
than a separate copy of the same idea -- see this script's own module
docstring for why that's deliberate, not an oversight.

## Status

Run for real on a Nebius L40S (2026-09-17): the real Llama-3-8B-Instruct
checkpoint, all 5 real tokenized prompts from `benchmarks/
speculative_decoding/prompts_tokenized.jsonl`.

```
[MATCH] medium-code-0001 (medium-code)
[MATCH] long-0002 (long)
[MATCH] medium-code-0003 (medium-code)
[MATCH] medium-code-0005 (medium-code)
[MATCH] medium-code-0006 (medium-code)
MATCH -- P/D-disaggregated generation agrees with single-engine and the
dense reference on every prompt.
```

Also verified alongside it: `model/tests/test_pd_disaggregation.py`'s 5
toy-config cases all pass, including the one that actually proves
block-layout independence through a real forward pass (a prefiller and a
decoder with different `block_size` *and* different `num_gpu_blocks`,
still byte-identical) -- not just asserted in `model/kv_cache.py`'s
comments. The full existing suite (`engine/tests/` + `model/tests/`,
185 cases) stayed green throughout, confirming none of this disturbed
anything the mechanism was built on top of.

The mechanism works, on this real checkpoint, exactly as designed. As
covered above, that's the whole claim this stage makes -- see "Why no
speed benchmark" for why a timing number isn't part of it.

## Why no speed benchmark

P/D disaggregation's entire real-world payoff is hardware isolation:
prefill is compute-bound, decode is memory-bandwidth-bound, and running
them on the same GPU makes them contend for both (see
`benchmarks/chunked_prefill/README.md` for this project's own concrete
numbers on that exact contention -- solved there a cheaper way, time-
slicing prefill via chunking on one GPU, rather than moving prefill to
separate hardware entirely).

This project runs on a single L40S (`terraform/terraform.tfvars`:
`instance_count = 1`) -- two `LLMEngine` instances here still share the
same SMs and the same memory bandwidth, so the isolation benefit this
technique exists for is structurally unavailable. A wall-clock
comparison would only measure the KV-transfer overhead this design adds,
on top of contention that's completely unchanged -- not a number worth
collecting, so no `measure_pd_speedup.py` exists alongside this script
the way speculative decoding got `measure_speedup.py`. See
`model/pd_disaggregation.py`'s own module docstring for the same point,
made where the mechanism itself lives.

## Files

- `verify_pd_correctness.py`: this script.