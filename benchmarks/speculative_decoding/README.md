# Speculative decoding: acceptance rate and speedup

Measures whether speculative decoding (a cheap Llama-3.2-1B draft
proposes K tokens per round, the real Llama-3-8B target verifies all K in
one forward pass and accepts the longest greedy-matching prefix, see
`model/draft_proposer.py` and `model/model_runner.py`'s `_accept_reject`)
is actually faster on the real checkpoint pair, not just correct (that's
`verify_speculative_correctness.py`'s job, in this same directory) --
acceptance rate, mean accepted tokens per target forward pass, wall-clock
inter-token latency (ITL), and tokens/s, swept over
`num_speculative_tokens` in `{0 (non-speculative baseline), 1, 2, 4, 8}`.

```bash
# once, if prompts_tokenized.jsonl doesn't exist yet:
sudo .venv/bin/python3 -m benchmarks.speculative_decoding.generate_tokenized_prompts

sudo .venv/bin/python3 -m benchmarks.speculative_decoding.measure_speedup
```

(`sudo` because the real checkpoints live under `/root/.cache/huggingface`
-- see `verify_speculative_correctness.py`'s module docstring for why.
Also `sudo docker stop vllm-server` first: it auto-starts on boot and
holds most of the L40S's GPU memory for its own serving otherwise.)

Writes `accept_summary.csv` (one row per `num_speculative_tokens` swept:
acceptance rate mean+stdev, mean accepted tokens per round mean+stdev,
derived target-forward-passes-per-output-token, ITL mean+stdev,
tokens/s mean+stdev), `accept_raw.csv` (one row per individual verify
round, `k>0` only), and `step_latency.csv` (one row per `engine.step()`
call) into this directory. `--repeats` defaults to 1 here, not the
5-with-first-discarded most other benchmarks in this repo use -- see the
script's own module docstring on why (real-weight reload cost per sweep
point, not toy-shaped).

## Status

Run for real on a Nebius L40S (2026-09-14, default sweep: 5 real
tokenized prompts, 80 max tokens each, `--repeats 1` with the first
discarded):

| num_speculative_tokens | acceptance_rate | mean_accepted_per_round | fwd_passes_per_token | itl_ms | tokens_per_second |
|-----------------------:|-----------------:|--------------------------:|-----------------------:|--------:|---------------------:|
| 0 (baseline)           | --                | 1.00                      | 1.00                   | 112.5   | 44.5                 |
| 1                      | 100.0%            | 1.77                      | 0.56                   | 224.6   | 37.7                 |
| 2                      | 89.9%             | 2.48                      | 0.40                   | 333.3   | 35.7                 |
| 4                      | 72.6%             | 3.38                      | 0.30                   | 531.5   | 29.2                 |
| 8                      | 53.7%             | 4.54                      | 0.22                   | 895.3   | 22.1                 |

**No configuration beats the non-speculative baseline.** Acceptance rate
degrades with K in exactly the expected shape -- each further draft
token is conditioned on an increasingly risky chain of prior guesses, so
100% at K=1 falling to 54% at K=8 is not a surprise. What is notable:
forward-pass efficiency improves monotonically with K (0.22 target
forward passes per output token at K=8, a genuine ~4.5x reduction from
the baseline's 1.00) -- the FLOP-level saving speculative decoding is
supposed to deliver is real and measured here -- but wall-clock ITL gets
*monotonically worse* over the same range, and tokens/s monotonically
lower. The forward-pass saving never translates into a wall-clock win.

Most likely cause, not separately profiled here (out of this stage's
scope, see the question this was checked against before writing this
section up): `DraftProposer.propose()`'s K proposed tokens come from K
*sequential* single-token decode steps on the draft model -- each one a
full `engine.step()`-shaped round-trip (Python scheduling, block
management, a real Triton kernel launch) -- and this repo has no CUDA
graphs or other per-step-overhead amortization (a deliberate, documented
scope choice throughout this project, see `engine/README.md`). At this
scale, that fixed per-step overhead on the draft side is real wall-clock
cost that the target-side saving doesn't outrun, and it compounds with K:
more proposed tokens means more sequential draft round-trips paid for
up front, even on a round that's later only partially accepted. This is
a known, real phenomenon in the speculative-decoding literature (the
technique's wall-clock payoff generally assumes an already-low-overhead
serving stack), not unique to this codebase -- but it's the actual,
measured conclusion for this implementation on this hardware: correct
(Stage F), but not faster here.

Raw data: `accept_summary.csv` / `accept_raw.csv` / `step_latency.csv` in
this directory.

## vLLM comparison

The natural follow-up to the finding above: is "no wall-clock speedup"
specific to lightserve's own unoptimized per-step overhead (no CUDA
graphs, plain Python scheduling), or does it hold even for a
production-optimized engine? Answered by running the *same* real
Llama-3.2-1B draft / Llama-3-8B target pair, on the same L40S, through
vLLM's own built-in speculative decoding instead.

**Methodology** — run manually this session, not via a committed
script (informal follow-up, not institutionalized the way
`measure_speedup.py` is): the same 5 real tokenized prompts as the table
above (`prompts_tokenized.jsonl`), same 80-token cap, sent concurrently
(one thread per prompt) to vLLM's `/v1/completions` endpoint. For each
`num_speculative_tokens` swept, the `vllm-server` container was
restarted fresh (`docker run ... vllm/vllm-openai:latest --model
meta-llama/Meta-Llama-3-8B-Instruct --speculative-config '{"method":
"draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct",
"num_speculative_tokens": K}'`, `HF_HUB_OFFLINE=1` since both checkpoints
are already cached, omitted entirely for the `k=0` baseline) — a launch-
time config can't change on a running server, and a fresh container also
means its Prometheus counters start at 0, so each sweep point's numbers
are read directly off one `/metrics` scrape at the end, no delta math
needed. vLLM exposes exactly the metrics needed natively, so no custom
client-side timing was required: `vllm:inter_token_latency_seconds_sum`
/ `_count` (ITL), `vllm:spec_decode_num_draft_tokens_total` /
`_num_accepted_tokens_total` / `_num_drafts_total` (acceptance rate =
accepted/draft; mean accepted per round = 1 + accepted/drafts, same
framing the table above uses, bonus token included), and
`vllm:generation_tokens_total` ÷ client-measured wall-clock for tokens/s.
vLLM 0.26.0 (confirmed via `python3 -c "import vllm; print(vllm.
__version__)"` inside the container before starting — well above the
0.10.0 minimum `--speculative-config`'s `draft_model` method needs).
Same `sudo docker stop vllm-server` GPU-freeing step as above, run
against a separately-named test container so the production one was
never touched; restarted at the end.

| num_speculative_tokens | acceptance_rate | mean_accepted_per_round | itl_ms | tokens_per_second |
|-----------------------:|-----------------:|--------------------------:|--------:|---------------------:|
| 0 (baseline)           | --                | 1.00                      | 22.0    | 208.0                |
| 1                       | 81.7%             | 1.82                      | 36.9    | 122.8                |
| 2                       | 63.6%             | 2.27                      | 108.3   | 65.7                 |
| 4                       | 58.1%             | 3.32                      | 220.8   | 48.5                 |
| 8                       | 40.2%             | 4.22                      | 190.9   | 47.3                 |

**Same qualitative shape as lightserve's own result, on a fully
production-optimized system.** vLLM's non-speculative baseline alone
(208 tok/s) is already ~4.7x faster than lightserve's (44.5 tok/s) --
real CUDA graphs and optimized kernels matter -- but *every* speculative
configuration is still slower than *that* baseline, and it gets worse as
K grows, same monotonic direction lightserve showed. This is a
meaningfully stronger conclusion than Stage G alone could support: **the
"no wall-clock speedup" finding is not specific to lightserve's
unoptimized per-step overhead** -- it reproduces on a production-grade
implementation with CUDA graphs and no Python-level scheduling overhead
to blame. The likely explanation shifts from "implementation overhead"
to workload shape: 5 concurrent, short (80-token) generations isn't
enough concurrent traffic for an 8x-smaller draft model's own cost to be
"free" relative to the batch it's competing against for the same GPU.

One side observation, not otherwise explained here: vLLM's acceptance
rate is consistently *lower* than lightserve's at every K (82% vs. 100%
at K=1, 40% vs. 54% at K=8) despite both using the identical checkpoints
and greedy sampling. Not dug into -- could be a real difference in
kernel-level numerics (this project already found one genuine
engine-vs-reference tie-breaking divergence at real-model scale, see
`verify_speculative_correctness.py`'s `EXCLUDED_IDS`), a difference in
exactly how each implementation seeds/advances the draft's own KV cache,
or something else entirely.

## Files

- `generate_tokenized_prompts.py` / `prompts_tokenized.jsonl`: real
  tokenized prompts (Stage F), reused here with `--max-tokens` overriding
  their capped values for a meaningful throughput reading.
- `verify_speculative_correctness.py`: the correctness counterpart to
  this speedup measurement -- byte-identical output is proven there, not
  re-checked here.
- `measure_speedup.py`: this benchmark.
