"""One-off diagnostic, not part of the regular benchmark suite (mirrors
benchmarks/chunked_prefill/verify_multi_chunk_correctness.py's shape):
checks that speculative decoding, through LLMEngine's real
speculative_config=SpeculativeConfig(...) constructor path, produces
output byte-identical to both plain (non-speculative) LLMEngine.generate()
and a dense from-scratch reference -- on the real Llama-3.2-1B draft /
Llama-3-8B target checkpoints, not TOY_CONFIG random weights.

Why this needed its own script rather than trusting model/tests/
test_llm_engine.py's TestSpeculative alone: that class's own toy-config
cases (added wiring speculative decoding into LLMEngine, see git log)
deliberately bypass speculative_config entirely -- a toy random model has
no HF checkpoint directory to load, so they hand-assign engine.
draft_proposer directly instead. That's the right call for testing the
verify/accept/rollback machinery cheaply, but it means LLMEngine.
__init__'s actual speculative_config branch (loading a real checkpoint via
model/hf_loader.py, sizing a second CacheConfig/PagedKVCache/ModelRunner
for the draft) had never run for real anywhere in this project before
this script.

No tokenizer in this repo by design (see model/README.md's scope notes),
so eos_token_id is deliberately left unset here -- generation relies
purely on each prompt's own max_tokens cap. That's not just a
simplification: it also re-exercises, on real checkpoints, the
speculative multi-token-overshoot truncation fix LLMEngine.step() got
this session (a toy-config test originally caught it; this is the first
time it's checked for real).

Setup (once): python3 -m benchmarks.speculative_decoding.generate_tokenized_prompts
Run manually on a CUDA GPU:
    python3 -m benchmarks.speculative_decoding.verify_speculative_correctness
"""
import json
import os
from pathlib import Path

import torch

from engine.config import CacheConfig, SchedulerConfig, SpeculativeConfig
from engine.request import SamplingParams
from model.hf_loader import load_hf_checkpoint
from model.llm_engine import LLMEngine
from model.minimal_llama import reference_llama_forward

PROMPTS_PATH = Path(__file__).parent / "prompts_tokenized.jsonl"
NUM_SPECULATIVE_TOKENS = 4
BLOCK_SIZE = 16

# Same glob-by-repo-dir-name resolution as model/tests/test_hf_loader.py's
# _find_snapshot_dir -- inlined rather than imported, same reasoning as
# generate_tokenized_prompts.py right next to this file.
_HF_HUB_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def _find_snapshot_dir(model_repo_dir_name):
    snapshots_dir = os.path.join(_HF_HUB_DIR, model_repo_dir_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for name in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, name)
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


def reference_generate(weights, config, prompt, max_tokens):
    """Same dense re-run-from-scratch-every-step reference model/tests/
    test_llm_engine.py's _reference_generate uses -- inlined here rather
    than imported, same reasoning as verify_multi_chunk_correctness.py.
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


def run_to_completion(engine, records):
    """Submits every record as its own request (its own max_tokens, unlike
    LLMEngine.generate()'s single shared sampling_params across a whole
    call) and steps until all are done -- same pattern model/tests/
    test_llm_engine.py's _run_to_completion uses, just batched over
    several requests at once instead of one.
    """
    requests = {
        r["id"]: engine.add_request(r["prompt"], sampling_params=SamplingParams(max_tokens=r["max_tokens"]),
                                     request_id=r["id"])
        for r in records
    }
    while engine.scheduler.has_unfinished_requests():
        engine.step()
    return {rid: req.output_token_ids for rid, req in requests.items()}


def main():
    records = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(
            f"No prompts found at {PROMPTS_PATH} -- run "
            "python3 -m benchmarks.speculative_decoding.generate_tokenized_prompts first."
        )

    target_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    draft_dir = _find_snapshot_dir("models--meta-llama--Llama-3.2-1B-Instruct")
    if target_dir is None or draft_dir is None:
        raise SystemExit(
            "Real checkpoints not found under "
            f"{_HF_HUB_DIR} -- see ~/.claude/plans/agile-rolling-gray.md's Context section."
        )

    target_config, target_weights = load_hf_checkpoint(target_dir, device="cuda")

    # Exact-fit block-count math (ceil-div token lengths, sum per request,
    # +1 block margin), same approach benchmarks/chunked_prefill/
    # measure_itl.py and benchmarks/prefix_caching/measure_ttft.py already
    # use for a real model -- not a GPU-bytes budget calc (engine/
    # config.py's own module docstring: deliberately out of scope here).
    # Same token-count math sizes the draft's own cache too; its bytes-
    # per-block are smaller (fewer layers/kv-heads) but the token-count
    # math doesn't care about bytes.
    num_gpu_blocks = sum(
        -(-(len(r["prompt"]) + r["max_tokens"]) // BLOCK_SIZE) for r in records
    ) + 1
    cache_config = CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=num_gpu_blocks)
    scheduler_config = SchedulerConfig(max_num_seqs=len(records), max_num_batched_tokens=2048)

    spec_config = SpeculativeConfig(
        draft_model_path=draft_dir,
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        draft_num_gpu_blocks=num_gpu_blocks,
    )
    spec_engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights,
                             device="cuda", speculative_config=spec_config)
    nonspec_engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights, device="cuda")

    spec_outputs = run_to_completion(spec_engine, records)
    nonspec_outputs = run_to_completion(nonspec_engine, records)

    all_match = True
    for r in records:
        expected = reference_generate(target_weights, target_config, r["prompt"], max_tokens=r["max_tokens"])
        spec_out = spec_outputs[r["id"]]
        nonspec_out = nonspec_outputs[r["id"]]
        ok = spec_out == nonspec_out == expected
        all_match &= ok
        status = "MATCH" if ok else "MISMATCH"
        print(f"[{status}] {r['id']} ({r['category']})")
        if not ok:
            print(f"  speculative: {spec_out}")
            print(f"  non-spec:    {nonspec_out}")
            print(f"  dense ref:   {expected}")

    if all_match:
        print("MATCH -- speculative decoding agrees with non-speculative and the dense reference on every prompt.")
    else:
        print("MISMATCH -- see above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
