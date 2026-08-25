"""Generates benchmarks/token_prompts.jsonl -- a length-matched companion to
benchmarks/baseline_prompts.jsonl, for load-testing lightserve's own
server/app.py instead of vLLM.

Why this needs to exist at all: baseline_prompts.jsonl's `prompt` is real
text, tokenized server-side by vLLM's real Llama-3 tokenizer. lightserve's
server has no tokenizer (see server/schemas.py's module docstring) -- its
`/v1/completions` takes `prompt` as token ids directly. So the same JSONL
can't drive both servers; this script derives a same-shape one instead: same
`id`/`category`/`max_tokens` per record (comparing the two engines at the
same category/output-length mix matters for a fair benchmark), but `prompt`
replaced with random token ids in [0, vocab_size) -- content-decoupled,
since lightserve's weights are random anyway (see model/minimal_llama.py's
module docstring), so real text would carry no more meaning here than
random ids do.

Per-record prompt *length* is chosen to match baseline_prompts.jsonl's, not
copied outright, since there's no tokenizer available here to measure each
text prompt's real Llama-3 token count (this repo's own triton/torch stack
has no macOS wheel, and pulling in a tokenizer just for this script would be
a new dependency for a one-line estimate). `_estimate_token_count` uses the
standard ~1.3-tokens-per-English-word rule of thumb instead -- an
approximation, not a tokenizer, called out here rather than presented as
exact.

Regenerate after editing this file or baseline_prompts.jsonl:
    python3 benchmarks/generate_token_prompts.py
"""
import json
import random
from pathlib import Path

IN_PATH = Path(__file__).parent / "baseline_prompts.jsonl"
OUT_PATH = Path(__file__).parent / "token_prompts.jsonl"

# meta-llama/Meta-Llama-3-8B-Instruct's real vocab size -- matches
# model/minimal_llama.py's llama3_8b_shape() default, so every generated id
# here is in-bounds for lightserve's default engine config too.
VOCAB_SIZE = 128_256

WORDS_PER_TOKEN = 1.3  # rough English words-per-token rule of thumb -- see module docstring
MIN_PROMPT_LEN = 4


def _estimate_token_count(text: str) -> int:
    return max(MIN_PROMPT_LEN, round(len(text.split()) * WORDS_PER_TOKEN))


def main():
    random.seed(42)  # same seed baseline_prompts.jsonl's own generator uses -- reproducible, not that it matters here
    records = []
    with IN_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src = json.loads(line)
            n = _estimate_token_count(src["prompt"])
            records.append({
                "id": src["id"],
                "category": src["category"],
                "prompt": [random.randrange(VOCAB_SIZE) for _ in range(n)],
                "max_tokens": src["max_tokens"],
            })

    with OUT_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
