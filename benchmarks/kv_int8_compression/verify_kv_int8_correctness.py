"""One-off diagnostic (mirrors benchmarks/pd_disaggregation/verify_pd_correctness.py's
shape): checks int8 KV cache compression (engine/config.py's CacheConfig.int8_kv,
model/kv_cache.py's PagedKVCache) on the real Llama-3-8B checkpoint.

Unlike every other verify_*_correctness.py script in this repo, this one is
NOT a byte-identical check -- int8_kv is lossy by construction (see
model/kv_cache.py's _quantize docstring), so byte-identical is the wrong bar.
Instead: run the same real tokenized prompts through a bf16-cache engine and
an int8-cache engine (same weights, same everything else) and report the
**top-1 token match rate** between the two -- how often int8 quantization
actually changed which token got sampled, not just how numerically different
the intermediate K/V values are.

Also reports the real per-block memory arithmetic (bf16 vs int8, including
the int8 path's fp32 scale-tensor overhead) for this checkpoint's actual
shape -- engine/config.py's own module docstring flags this arithmetic as
"deliberately not [implemented] here"; this script is the first place in
this repo that actually computes it, since int8_kv's whole motivation is the
capacity/bandwidth this arithmetic quantifies.

No --min-match-rate default: this is a report, not a pass/fail gate, since
deciding "how much divergence is acceptable" is a real design decision, not
something to silently bake in as a hardcoded threshold. Pass --min-match-rate
to turn it into a gate (e.g. for CI) once that decision is actually made.

Reuses benchmarks/speculative_decoding/prompts_tokenized.jsonl rather than a
separate copy, same reasoning as every other verify_*_correctness.py script
in this repo.

Run manually on a CUDA GPU:
    python3 -m benchmarks.kv_int8_compression.verify_kv_int8_correctness
"""
import argparse
import json
import os
from pathlib import Path

from engine.config import CacheConfig, SchedulerConfig
from engine.request import SamplingParams
from model.hf_loader import load_hf_checkpoint
from model.llm_engine import LLMEngine

PROMPTS_PATH = Path(__file__).parent.parent / "speculative_decoding" / "prompts_tokenized.jsonl"
BLOCK_SIZE = 16

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


def run_to_completion(engine, records):
    """Same pattern as verify_pd_correctness.py's run_single_engine: submit
    every record, step until nothing's left running, read outputs back off
    the Request objects.
    """
    requests = {
        r["id"]: engine.add_request(r["prompt"], sampling_params=SamplingParams(max_tokens=r["max_tokens"]),
                                     request_id=r["id"])
        for r in records
    }
    while engine.scheduler.has_unfinished_requests():
        engine.step()
    return {rid: req.output_token_ids for rid, req in requests.items()}


def token_match_rate(a: list, b: list) -> tuple:
    """Position-by-position top-1 match rate over the shorter of the two
    sequences -- early divergence can also change *when* generation stops
    (a different eos/length outcome), so comparing only up to the shorter
    length is the honest choice: comparing past that would be comparing a
    real token against nothing. Returns (matches, compared, first_diverge
    or None).
    """
    n = min(len(a), len(b))
    matches = 0
    first_diverge = None
    for i in range(n):
        if a[i] == b[i]:
            matches += 1
        elif first_diverge is None:
            first_diverge = i
    return matches, n, first_diverge


def report_memory_arithmetic(model_config, block_size: int) -> None:
    """The physical-GPU-memory math engine/config.py's own module docstring
    flags as deliberately not implemented anywhere in this repo -- computed
    here for the first time, since int8_kv's whole motivation is exactly
    this number. Per-block bytes: block_size tokens * 2 (K and V) *
    n_layers * num_kv_heads * head_dim * dtype_bytes, plus (int8 only) the
    fp32 scale tensor's own overhead -- one scalar per token per KV head,
    not per element, so head_dim doesn't multiply that term.
    """
    n_layers = model_config.n_layers
    num_kv_heads = model_config.num_kv_heads
    head_dim = model_config.head_dim

    bf16_bytes_per_block = block_size * 2 * n_layers * num_kv_heads * head_dim * 2
    int8_storage_bytes_per_block = block_size * 2 * n_layers * num_kv_heads * head_dim * 1
    int8_scale_bytes_per_block = block_size * 2 * n_layers * num_kv_heads * 4  # fp32 scale, no head_dim factor
    int8_bytes_per_block = int8_storage_bytes_per_block + int8_scale_bytes_per_block
    capacity_multiple = bf16_bytes_per_block / int8_bytes_per_block

    print("\n--- Memory arithmetic (per KV-cache block, this checkpoint's real shape) ---")
    print(f"n_layers={n_layers} num_kv_heads={num_kv_heads} head_dim={head_dim} block_size={block_size}")
    print(f"bf16:  {bf16_bytes_per_block:,} bytes/block")
    print(f"int8:  {int8_bytes_per_block:,} bytes/block "
          f"({int8_storage_bytes_per_block:,} storage + {int8_scale_bytes_per_block:,} fp32 scale overhead)")
    print(f"Same GPU memory budget fits {capacity_multiple:.2f}x as many blocks under int8_kv "
          f"(not quite 2x -- the scale tensors are real, if small, overhead).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-match-rate", type=float, default=None,
                     help="If set, exit 1 when the overall token match rate falls below this "
                          "(e.g. 0.95). Unset by default -- see module docstring on why this "
                          "is a report, not a pass/fail gate, until that bar is actually decided.")
    args = ap.parse_args()

    records = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No prompts found at {PROMPTS_PATH}.")

    target_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    if target_dir is None:
        raise SystemExit(f"Real Llama-3-8B-Instruct checkpoint not found under {_HF_HUB_DIR}.")

    target_config, target_weights = load_hf_checkpoint(target_dir, device="cuda")

    # Same exact-fit block-count math every real-checkpoint script in this
    # repo uses (see verify_pd_correctness.py's own comment on this) --
    # doubled headroom isn't needed since bf16 and int8 engines each get
    # their own separate PagedKVCache sized off the same records.
    num_gpu_blocks = sum(-(-(len(r["prompt"]) + r["max_tokens"]) // BLOCK_SIZE) for r in records) + 1
    scheduler_config = SchedulerConfig(max_num_seqs=len(records), max_num_batched_tokens=2048)

    bf16_cache_config = CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=num_gpu_blocks, int8_kv=False)
    int8_cache_config = CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=num_gpu_blocks, int8_kv=True)

    bf16_engine = LLMEngine(bf16_cache_config, scheduler_config, target_config,
                             weights=target_weights, device="cuda")
    int8_engine = LLMEngine(int8_cache_config, scheduler_config, target_config,
                             weights=target_weights, device="cuda")

    bf16_outputs = run_to_completion(bf16_engine, records)
    int8_outputs = run_to_completion(int8_engine, records)

    total_matches, total_compared = 0, 0
    for r in records:
        bf16_out = bf16_outputs[r["id"]]
        int8_out = int8_outputs[r["id"]]
        matches, compared, first_diverge = token_match_rate(bf16_out, int8_out)
        total_matches += matches
        total_compared += compared
        rate = matches / compared if compared else float("nan")
        diverge_str = f"first divergence at token {first_diverge}" if first_diverge is not None else "no divergence"
        len_str = "" if len(bf16_out) == len(int8_out) else f" (bf16 len={len(bf16_out)}, int8 len={len(int8_out)})"
        print(f"[{r['id']} ({r['category']})] match_rate={rate:.1%} ({matches}/{compared}) {diverge_str}{len_str}")

    overall_rate = total_matches / total_compared if total_compared else float("nan")
    print(f"\nOverall top-1 token match rate: {overall_rate:.1%} ({total_matches}/{total_compared})")

    report_memory_arithmetic(target_config, BLOCK_SIZE)

    if args.min_match_rate is not None and overall_rate < args.min_match_rate:
        print(f"\nBelow --min-match-rate={args.min_match_rate:.1%} -- treating as a failure.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
