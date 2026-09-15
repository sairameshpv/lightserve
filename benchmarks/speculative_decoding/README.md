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

## Files

- `generate_tokenized_prompts.py` / `prompts_tokenized.jsonl`: real
  tokenized prompts (Stage F), reused here with `--max-tokens` overriding
  their capped values for a meaningful throughput reading.
- `verify_speculative_correctness.py`: the correctness counterpart to
  this speedup measurement -- byte-identical output is proven there, not
  re-checked here.
- `measure_speedup.py`: this benchmark.
