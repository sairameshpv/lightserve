"""One-off diagnostic, not part of the regular benchmark suite: checks
whether a genuinely multi-step chunked prefill (a prompt long enough, and
a chunk size small enough relative to it, that Scheduler._schedule_running
must resume it across 2+ steps before it's fully prefilled) produces the
same output as LLMEngine.generate() would for a prompt that fits in one
step -- the same "matches a dense reference" property
model/tests/test_llm_engine.py's TestGenerate already checks, just never
with a prompt actually long enough to force multi-step chunking (every
prompt there is 3-5 tokens against a 64-token default budget).

Why this needed checking: Request.get_num_new_tokens() is `get_len() -
num_computed_tokens`, and get_len() is `len(prompt_token_ids) +
len(output_token_ids)` -- but model/model_runner.py's execute_model()
samples and appends a token to output_token_ids for *every* scheduled
request every step, including one still mid-prefill (is_prefill() still
True), not only once it's actually done. Traced by hand: once a
multi-chunk prefill's chunk size doesn't evenly divide the prompt length,
the accumulated (still-mid-prefill) samples inflate get_num_new_tokens()
right at the tail chunk, in principle pushing that chunk's token slice
past the real prompt boundary into those garbage samples.

Run manually on a CUDA GPU:
    python3 -m benchmarks.chunked_prefill.verify_multi_chunk_correctness
"""
from dataclasses import replace

import torch

from engine.config import CacheConfig, SchedulerConfig
from engine.request import SamplingParams
from model.llm_engine import LLMEngine
from model.minimal_llama import TOY_CONFIG, init_weights, reference_llama_forward


def reference_generate(weights, config, prompt, max_tokens):
    """Same dense re-run-from-scratch-every-step reference
    model/tests/test_llm_engine.py's _reference_generate uses -- inlined
    here rather than imported, so this script has no dependency on pytest
    test-file internals.
    """
    tokens = list(prompt)
    generated = []
    for _ in range(max_tokens):
        input_ids = torch.tensor([tokens], device="cuda")
        logits = reference_llama_forward(weights, config, input_ids, causal=True)
        next_tok = int(logits[0, -1].argmax().item())
        tokens.append(next_tok)
        generated.append(next_tok)
    return generated


def main():
    torch.manual_seed(0)
    config = replace(TOY_CONFIG, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)

    # prompt_len=10, chunk=3: 10/3 doesn't divide evenly (3,3,3,1 real
    # tokens across 4 admission/continuation steps) -- exactly the
    # boundary case traced by hand: by the 4th chunk, 3 mid-prefill
    # samples have already been appended to output_token_ids, inflating
    # get_num_new_tokens() past the 1 real remaining prompt token.
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_tokens = 5  # continues well past full prefill into real decode

    cache_config = CacheConfig(block_size=4, num_gpu_blocks=64)
    scheduler_config = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=3)
    engine = LLMEngine(cache_config, scheduler_config, config, weights=weights, device="cuda")

    [output] = engine.generate([prompt], sampling_params=SamplingParams(max_tokens=max_tokens))
    actual = output.output_token_ids

    torch.manual_seed(0)
    weights_ref = init_weights(config, device="cuda", seed=0)
    expected = reference_generate(weights_ref, config, prompt, max_tokens=max_tokens)

    print(f"chunked (max_num_batched_tokens=3): {actual}")
    print(f"dense reference:                    {expected}")
    if actual == expected:
        print("MATCH -- multi-step chunked prefill agrees with the dense reference.")
    else:
        print("MISMATCH -- multi-step chunked prefill diverges from the dense reference.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()