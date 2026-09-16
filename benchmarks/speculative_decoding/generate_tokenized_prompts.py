"""Generates benchmarks/speculative_decoding/prompts_tokenized.jsonl -- a
small, *really* tokenized subset of benchmarks/baseline_prompts.jsonl, for
verify_speculative_correctness.py.

Different from benchmarks/generate_token_prompts.py's random-id approach
on purpose: that script exists because lightserve's own random weights
make real text meaningless anyway (see its module docstring). Speculative
decoding's draft/target correlation is the whole point being verified
here, so random ids -- which no real tokenizer would ever produce as a
coherent prompt -- would give a degenerate, uninformative accept/reject
pattern. Needs the real tokenizer (via the `transformers` package, not a
dependency the rest of this repo carries -- see model/README.md's scope
notes on staying tokenizer-free) and the real 8B checkpoint's tokenizer
files (already on disk alongside its weights, downloaded together).

Only a handful of prompts, with max_tokens capped well below baseline_
prompts.jsonl's originals (up to 220) -- this feeds a correctness check,
not a throughput sweep (that's Stage G's job), so it only needs enough
rounds to exercise partial/full acceptance and rollback for real, not a
representative workload.

Regenerate after editing this file or baseline_prompts.jsonl:
    python3 -m benchmarks.speculative_decoding.generate_tokenized_prompts

--categories (comma-separated category:count pairs, e.g. "medium-code:5")
selects exactly that many of each named category instead of the default
"first NUM_PROMPTS records regardless of category" -- for building a
domain-isolated prompt set (see benchmarks/speculative_decoding/
README.md's "Speculative tuning" section) without disturbing the default
prompts_tokenized.jsonl, which existing correctness/speedup runs already
depend on. Pair with --out to write somewhere else instead of overwriting
it:
    python3 -m benchmarks.speculative_decoding.generate_tokenized_prompts \\
        --categories medium-code:5 --out prompts_tokenized_code.jsonl
"""
import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

IN_PATH = Path(__file__).parent.parent / "baseline_prompts.jsonl"
OUT_PATH = Path(__file__).parent / "prompts_tokenized.jsonl"

NUM_PROMPTS = 5
MAX_TOKENS = 24

# medium-reasoning-0004 hits a genuine exact logit tie (two tokens scoring
# identically) at generation step 1 on the real 8B checkpoint -- greedy
# argmax's tie-break then depends on sub-ULP numerical differences between
# the engine's Triton kernel path and a dense PyTorch reference, which can
# disagree on which one wins. Confirmed not a functional bug (reproduces
# with speculative decoding entirely out of the picture -- a solo, non-
# speculative engine run alone still disagrees with the dense reference at
# this exact spot) and not batching-related (same result run alone or
# batched). Excluded here because it's not useful for a script whose whole
# point is a strict byte-identical check, not because anything is broken.
EXCLUDED_IDS = {"medium-reasoning-0004"}

# Same glob-by-repo-dir-name resolution as model/tests/test_hf_loader.py's
# _find_snapshot_dir -- reimplemented inline here rather than imported
# from a test module, matching benchmarks/chunked_prefill/verify_multi_
# chunk_correctness.py's precedent of inlining small one-off pieces.
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


def _select_sources(categories_arg: str) -> list:
    """Reads every non-excluded source record once (baseline_prompts.jsonl
    is small, ~1000 lines -- loading it fully is simpler than an
    interleaved single pass). Unset categories_arg: first NUM_PROMPTS in
    file order, today's original behavior. Set: exactly count of each
    named category, in file order within that category.
    """
    all_sources = []
    with IN_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src = json.loads(line)
            if src["id"] not in EXCLUDED_IDS:
                all_sources.append(src)

    if not categories_arg:
        return all_sources[:NUM_PROMPTS]

    selected = []
    for part in categories_arg.split(","):
        category, count_str = part.split(":")
        count = int(count_str)
        matching = [s for s in all_sources if s["category"] == category][:count]
        if len(matching) < count:
            raise SystemExit(f"Only {len(matching)} non-excluded '{category}' records available, "
                              f"needed {count}")
        selected.extend(matching)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", default=None,
                     help="Comma-separated category:count pairs (e.g. medium-code:5,medium-reasoning:5); "
                          "unset keeps the default first-NUM_PROMPTS-in-file-order behavior")
    ap.add_argument("--out", default=str(OUT_PATH), help="Output path; defaults to prompts_tokenized.jsonl")
    args = ap.parse_args()
    out_path = Path(args.out)

    checkpoint_dir = _find_snapshot_dir("models--meta-llama--Meta-Llama-3-8B-Instruct")
    if checkpoint_dir is None:
        raise SystemExit(
            "Real Llama-3-8B-Instruct checkpoint not found under "
            f"{_HF_HUB_DIR} -- see ~/.claude/plans/agile-rolling-gray.md's Context section."
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    records = []
    for src in _select_sources(args.categories):
        token_ids = tokenizer.encode(src["prompt"])  # real tokenization, BOS included by default
        records.append({
            "id": src["id"],
            "category": src["category"],
            "prompt": token_ids,
            "max_tokens": min(src["max_tokens"], MAX_TOKENS),
        })

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
