"""Throwaway diagnostic, not part of the benchmark suite -- investigates
the one real divergence verify_kv_int8_correctness.py found (prompt
medium-code-0006, diverges at token index 1). Monkeypatches
ModelRunner._sample to capture pre-argmax logits for one prompt, run
through both a bf16 and an int8 engine, and reports the top-2 logit gap
at the step that produced the diverging token -- distinguishes "a
genuine near-tie flipped by quantization noise" (small gap) from "an
actual bug" (a large, suspicious gap) before trusting the correctness
script's headline number.
"""
import json
import os
from pathlib import Path

import torch

from engine.config import CacheConfig, SchedulerConfig
from engine.request import SamplingParams
from model.hf_loader import load_hf_checkpoint
from model.llm_engine import LLMEngine

PROMPTS_PATH = Path(__file__).parent.parent / "speculative_decoding" / "prompts_tokenized.jsonl"
_HF_HUB_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def _find_snapshot_dir(model_repo_dir_name):
    snapshots_dir = os.path.join(_HF_HUB_DIR, model_repo_dir_name, "snapshots")
    for name in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, name)
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None


def run_and_capture(engine, record, max_tokens):
    captured = []
    original_sample = engine.model_runner._sample

    def hooked_sample(logits):
        captured.append(logits.detach().clone())
        return original_sample(logits)

    engine.model_runner._sample = hooked_sample
    req = engine.add_request(record["prompt"], sampling_params=SamplingParams(max_tokens=max_tokens),
                              request_id=record["id"])
    while engine.scheduler.has_unfinished_requests():
        engine.step()
    return req.output_token_ids, captured


def top2_gap(logits_row: torch.Tensor):
    top2 = torch.topk(logits_row.float(), k=2)
    top1_val, top2_val = top2.values[0].item(), top2.values[1].item()
    top1_tok, top2_tok = top2.indices[0].item(), top2.indices[1].item()
    return top1_tok, top1_val, top2_tok, top2_val, top1_val - top2_val


def main():
    records = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["id"] == "medium-code-0006":
                    records.append(r)
    assert len(records) == 1, f"expected exactly one matching record, got {len(records)}"
    record = records[0]

    target_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    target_config, target_weights = load_hf_checkpoint(target_dir, device="cuda")

    num_gpu_blocks = -(-(len(record["prompt"]) + record["max_tokens"]) // 16) + 1
    scheduler_config = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4096)

    bf16_engine = LLMEngine(CacheConfig(block_size=16, num_gpu_blocks=num_gpu_blocks, int8_kv=False),
                             scheduler_config, target_config, weights=target_weights, device="cuda")
    int8_engine = LLMEngine(CacheConfig(block_size=16, num_gpu_blocks=num_gpu_blocks, int8_kv=True),
                             scheduler_config, target_config, weights=target_weights, device="cuda")

    bf16_tokens, bf16_logits = run_and_capture(bf16_engine, record, max_tokens=2)
    int8_tokens, int8_logits = run_and_capture(int8_engine, record, max_tokens=2)

    print(f"bf16 tokens: {bf16_tokens}  ({len(bf16_logits)} _sample calls captured)")
    print(f"int8 tokens: {int8_tokens}  ({len(int8_logits)} _sample calls captured)")

    # Don't assume call index N corresponds to output token N -- _sample may
    # be called on steps whose result never gets appended (e.g. mid-chunked-
    # prefill). Print every captured call's own argmax so the real
    # alignment against output_token_ids is visible, not assumed.
    print("\nEvery captured _sample call's own argmax, in order:")
    for step_idx, logits in enumerate(bf16_logits):
        print(f"  bf16 call {step_idx}: argmax={logits[0].argmax().item()}")
    for step_idx, logits in enumerate(int8_logits):
        print(f"  int8 call {step_idx}: argmax={logits[0].argmax().item()}")

    for step_idx in range(min(len(bf16_logits), len(int8_logits))):
        b_top1, b_val, b_top2, b_val2, b_gap = top2_gap(bf16_logits[step_idx][0])
        i_top1, i_val, i_top2, i_val2, i_gap = top2_gap(int8_logits[step_idx][0])
        agree = "MATCH" if b_top1 == i_top1 else "DIVERGE"
        print(f"\nstep {step_idx} [{agree}]:")
        print(f"  bf16: top1={b_top1} (logit={b_val:.4f}) top2={b_top2} (logit={b_val2:.4f}) gap={b_gap:.4f}")
        print(f"  int8: top1={i_top1} (logit={i_val:.4f}) top2={i_top2} (logit={i_val2:.4f}) gap={i_gap:.4f}")
        # How far did int8's logit for bf16's chosen token move?
        b_choice_int8_logit = int8_logits[step_idx][0][b_top1].item()
        print(f"  int8's logit for bf16's top-1 token ({b_top1}): {b_choice_int8_logit:.4f} "
              f"(vs int8's own top-1 logit {i_val:.4f}, delta={i_val - b_choice_int8_logit:.4f})")


if __name__ == "__main__":
    main()
