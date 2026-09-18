# P/D (prefill/decode) disaggregation

Proves the *mechanism* for prefill/decode disaggregation -- running a
request's prefill on one `LLMEngine` and its decode on a different one,
handing the prompt's KV cache across instead of recomputing it (see
`model/pd_disaggregation.py`) -- correctly, on this project's real
Llama-3-8B checkpoint. On its own, this script does **not**, and
structurally cannot usefully, measure a speedup: see "Why no speed
benchmark" below. A real 2-GPU speedup measurement was run separately,
once real hardware was available -- see "Real 2-GPU speedup".

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

This reasoning holds for *this* script specifically, which is
correctness-only by design. A real 2-GPU measurement was run
separately once real hardware was available -- see "Real 2-GPU
speedup" below for what it found, and its own limits.

## Real 2-GPU speedup

A separate script, `measure_pd_real_speedup.py`, was run on two real,
separate L40S nodes (2026-09-18) -- prefill on one, decode on the
other, talking over real HTTP via `server/pd_role_server.py` -- to
measure whether the hardware-isolation benefit above is real. Workload:
4 concurrent decode requests settled to steady state, then one
2048-token prefill injected mid-stream (same settle-then-inject shape
`benchmarks/chunked_prefill/measure_itl.py` uses).

| Condition | decode ITL baseline | decode ITL during injected prefill | worst single-step stall |
|---|---|---|---|
| 1 L40S (prefill+decode share a GPU) | 323.3ms | 459.8ms (+42%) | 2197ms |
| 2 L40S (prefill and decode on separate GPUs) | 98.6ms | 98.2ms (~0%) | 98.8ms |

On one shared GPU, the injected prefill measurably stalls decode --
mean per-token latency jumps 42%, worst-case wait balloons past 2
seconds. Move prefill to a second, physically separate GPU and that
disruption vanishes: 98.6ms -> 98.2ms is noise, not a measurable
effect. Real hardware isolation eliminates the contention this
project's own chunked-prefill numbers first showed on one GPU.

**Two things this result does, and does not, show.** First, the
absolute baseline levels (323ms vs. 98.6ms) aren't a clean comparison
-- node 0's 1-GPU run shared its GPU with an idle-but-resident prefill
role-server process left over from setting up the 2-GPU condition, an
uncontrolled confound. Only the *relative* baseline-to-disrupted delta
within each condition is trusted here (that background load is roughly
constant across both phases of the same run, so it shouldn't skew the
delta even though it skews the absolute numbers).

Second, and more fundamentally: this compares 1 GPU against 2 GPUs, not
"disaggregation" against an equal-hardware alternative. It proves the
isolation *mechanism* -- contention that exists on one shared GPU
disappears on separate hardware -- but does not show disaggregation is
a *better use of a 2nd GPU* than the obvious alternative: a 2nd
independent monolithic replica, load-balanced across both (the
production "two-pool" question). That needs matched total hardware and
aggregate throughput across the whole deployment, not one node's decode
ITL, and wasn't measured here.

## Files

- `verify_pd_correctness.py`: this script.
- `measure_pd_real_speedup.py`: the real 2-GPU benchmark (see "Real
  2-GPU speedup" above). Needs two already-running `server.pd_role_server`
  processes, one `--role prefill`, one `--role decode` -- see its own
  module docstring for the exact commands and `--stage` flag.