"""One-off diagnostic, not part of the regular benchmark suite (mirrors
benchmarks/speculative_decoding/verify_speculative_correctness.py's shape):
checks that P/D (prefill/decode) disaggregation -- model/pd_disaggregation.py's
prefill() on one LLMEngine + resume_decode() on a *different* one -- produces
output byte-identical to both an ordinary single-engine run and a dense
from-scratch reference, on the real Llama-3-8B checkpoint.

Why this needed its own script rather than trusting model/tests/
test_pd_disaggregation.py alone: that file's toy-config tests (TOY_CONFIG,
random weights) prove the transfer mechanism -- including the block-layout-
independence claim, via two engines with deliberately different block_size/
num_gpu_blocks -- cheaply and thoroughly. What they don't touch is a real
bf16 checkpoint's actual weights: real GQA (num_kv_heads=8, not MHA), real
head_dim=128, real n_layers=32. This is the first time this project moves
real KV bytes between two independent PagedKVCache instances.

Only one model is needed here, unlike speculative decoding's draft+target
pair -- P/D disaggregation splits *where* one model's own forward passes
run, not which model runs them. Both prefill_engine and decode_engine load
the same real Llama-3-8B-Instruct checkpoint (weights loaded once, shared
by reference the same way verify_speculative_correctness.py's spec_engine/
nonspec_engine already share target_weights -- inference-time weight
tensors are read-only, so this is safe).

Reuses benchmarks/speculative_decoding/prompts_tokenized.jsonl rather than
a second copy -- those are just real tokenized prompts, domain-agnostic to
what's being verified against them; no reason to duplicate the file or its
generator script for this stage.

No tokenizer in this repo by design (see model/README.md's scope notes),
so eos_token_id is deliberately left unset here -- generation relies
purely on each prompt's own max_tokens cap, same reasoning
verify_speculative_correctness.py uses.

Correctness only, deliberately -- no measure_pd_speedup.py alongside this
script the way speculative decoding got measure_speedup.py. This project
runs on a single L40S (see model/pd_disaggregation.py's module docstring);
P/D disaggregation's entire real-world payoff is running prefill and
decode on genuinely separate hardware, which one GPU cannot provide. A
timing comparison here would only measure added CPU-transfer overhead
against no isolation benefit -- not a useful number, so it isn't collected.

Run manually on a CUDA GPU:
    python3 -m benchmarks.pd_disaggregation.verify_pd_correctness
"""
import json
import os
from pathlib import Path

import torch

from engine.config import CacheConfig, SchedulerConfig
from engine.request import SamplingParams
from model.hf_loader import load_hf_checkpoint
from model.llm_engine import LLMEngine
from model.minimal_llama import reference_llama_forward
from model.pd_disaggregation import generate_disaggregated

PROMPTS_PATH = Path(__file__).parent.parent / "speculative_decoding" / "prompts_tokenized.jsonl"
BLOCK_SIZE = 16

# Same glob-by-repo-dir-name resolution as model/tests/test_hf_loader.py's
# _find_snapshot_dir -- inlined rather than imported, same reasoning as
# every other benchmarks/*/verify_*_correctness.py script in this repo.
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
    than imported, same reasoning as every verify_*_correctness.py script.
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


def run_disaggregated(prefill_engine, decode_engine, records):
    """One request at a time, start to finish, through the P/D split --
    unlike run_single_engine below, deliberately sequential rather than
    all `records` concurrently. This script's job is per-request transfer
    correctness, not throughput under concurrency (that's explicitly out
    of scope here, see module docstring) -- sequential is simplest and
    fully exercises the mechanism for every prompt, while still reusing
    the same two engines across all of them (so block allocation/freeing
    across multiple requests over time gets exercised too, not just one).
    """
    outputs = {}
    for r in records:
        result = generate_disaggregated(
            prefill_engine, decode_engine, r["prompt"],
            sampling_params=SamplingParams(max_tokens=r["max_tokens"]), request_id=r["id"],
        )
        outputs[r["id"]] = result.output_token_ids
    return outputs


def run_single_engine(engine, records):
    """The non-disaggregated baseline -- every record submitted to one
    ordinary engine and stepped to completion together, same pattern
    verify_speculative_correctness.py's run_to_completion uses.
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
    if target_dir is None:
        raise SystemExit(
            f"Real Llama-3-8B-Instruct checkpoint not found under {_HF_HUB_DIR} -- "
            "see ~/.claude/plans/agile-rolling-gray.md's Context section."
        )

    target_config, target_weights = load_hf_checkpoint(target_dir, device="cuda")

    # Exact-fit block-count math (ceil-div token lengths, sum per request,
    # +1 block margin), same approach every real-checkpoint benchmark
    # script in this repo uses -- not a GPU-bytes budget calc (engine/
    # config.py's own module docstring: deliberately out of scope here).
    # Shared by all three engines below; one BLOCK_SIZE, not a deliberately
    # differing pair -- that structural claim is already proven cheaply by
    # model/tests/test_pd_disaggregation.py's toy-config tests, this
    # script's job is real-checkpoint correctness, not re-proving it.
    num_gpu_blocks = sum(
        -(-(len(r["prompt"]) + r["max_tokens"]) // BLOCK_SIZE) for r in records
    ) + 1
    cache_config = CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=num_gpu_blocks)
    scheduler_config = SchedulerConfig(max_num_seqs=len(records), max_num_batched_tokens=2048)

    prefill_engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights, device="cuda")
    decode_engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights, device="cuda")
    single_engine = LLMEngine(cache_config, scheduler_config, target_config, weights=target_weights, device="cuda")

    disagg_outputs = run_disaggregated(prefill_engine, decode_engine, records)
    single_outputs = run_single_engine(single_engine, records)

    all_match = True
    for r in records:
        expected = reference_generate(target_weights, target_config, r["prompt"], max_tokens=r["max_tokens"])
        disagg_out = disagg_outputs[r["id"]]
        single_out = single_outputs[r["id"]]
        ok = disagg_out == single_out == expected
        all_match &= ok
        status = "MATCH" if ok else "MISMATCH"
        print(f"[{status}] {r['id']} ({r['category']})")
        if not ok:
            print(f"  disaggregated: {disagg_out}")
            print(f"  single-engine: {single_out}")
            print(f"  dense ref:     {expected}")

    if all_match:
        print("MATCH -- P/D-disaggregated generation agrees with single-engine and the dense reference "
              "on every prompt.")
    else:
        print("MISMATCH -- see above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()