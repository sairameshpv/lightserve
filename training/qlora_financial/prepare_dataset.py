"""Normalizes four real, document-grounded financial-QA datasets into
one common chat-format instruction set for QLoRA fine-tuning of
meta-llama/Meta-Llama-3-8B-Instruct, per
~/.claude/plans/floating-squishing-sonnet.md's Design section.

Deliberately local-machine-testable: only needs `datasets` (to pull the
four HF sources) and `transformers` (for the tokenizer, length-audit
only -- no torch/GPU/bitsandbytes required at this stage). Real schema
assumptions below were based on HF dataset-card *descriptions*, not a
live load -- this script's own first real run is what actually confirms
them; every per-source normalizer below fails loudly (asserts on the
raw first example) rather than silently producing wrong text if a field
name doesn't match what the dataset card described.

The four sources, and the one real design decision made per source
(see the plan's Design section for the full reasoning, not repeated
here):
- virattt/financial-qa-10K: plain question/answer/context, used as-is.
- FinGPT/fingpt-convfinqa: already instruction/input/output-formatted;
  its output stays a bare number, not reformatted into a fuller
  sentence (that would mean fabricating text not in the source data).
- FinQA (canonical, with pre_text/post_text/table/question/answers/
  program): the gold program becomes an explicit step-by-step
  calculation section before the final answer -- real CoT signal, not
  discarded. Falls back to plain question->answer for any row whose
  program can't be parsed, rather than emitting wrong reasoning text.
- next-tat/TAT-QA: table + linked paragraphs become the context;
  `derivation` gets the same step-before-answer treatment as FinQA's
  program, for consistency between the two harder-reasoning sources.

Run:
    python3 -m training.qlora_financial.prepare_dataset --out-dir training/qlora_financial/data
"""
import argparse
import ast
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer questions about "
    "company financial documents accurately and concisely, using only "
    "the information given in the context."
)

SEED = 0


def _make_example(user_content: str, assistant_content: str, source: str, group: str) -> dict:
    # group: which source document this example is about -- split() keeps
    # every example sharing a group in the same split (no test leakage).
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": source,
        "group": group,
    }


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def render_table(table) -> str:
    """table: a 2D list (rows of cells). Renders as a plain-text grid --
    not real markdown table syntax, since financial table cells often
    contain '|' and '-' themselves (e.g. date ranges, negative numbers
    in parens) which would corrupt markdown table parsing downstream.
    """
    if not table:
        return ""
    rows = ["\t".join(str(cell) for cell in row) for row in table]
    return "\n".join(rows)


# -- FinQA program DSL -> readable steps ------------------------------
#
# FinQA's program field is a comma-separated sequence of
# `operation(arg1, arg2)` calls, args are either raw numbers, `#N`
# referencing step N's own result (0-indexed), or a `const_N` /
# `const_mN` literal-constant token (`m` prefix = negative, e.g.
# const_m1 = -1) -- confirmed against real loaded rows from
# wandb/finqa-data-processed (every const_ suffix that actually
# appears in the data was enumerated directly: 1,2,3,4,5,6,7,8,9,10,
# 100,1000,100000,1000000,m1 -- all plain digits or the m-negative
# form, no other convention seen). _program_to_steps returns None on
# anything it still can't parse, and the caller falls back to plain
# question->answer for that row rather than emit a wrong reasoning
# chain.
_OP_RE = re.compile(r"(\w+)\(([^)]*)\)")
_CONST_RE = re.compile(r"^const_(m?\d+)$")
_OP_WORDS = {
    "add": "add", "subtract": "subtract", "multiply": "multiply",
    "divide": "divide", "exp": "raise to the power of",
    "greater": "compare (greater than)", "table_max": "take the max of",
    "table_min": "take the min of", "table_sum": "sum",
    "table_average": "average",
}


def _program_to_steps(program: str, results: list = None) -> list:
    if not program:
        return None
    calls = _OP_RE.findall(program)
    if not calls:
        return None
    steps = []
    for i, (op, args_str) in enumerate(calls):
        if op not in _OP_WORDS:
            return None  # unknown operator -- don't guess, fall back
        args = [a.strip() for a in args_str.split(",")]
        resolved = []
        for a in args:
            ref_match = re.match(r"^#(\d+)$", a)
            const_match = _CONST_RE.match(a)
            if ref_match:
                ref = int(ref_match.group(1))
                if ref >= i:
                    return None  # forward/self reference -- malformed
                resolved.append(f"the result of step {ref + 1}")
            elif const_match:
                suffix = const_match.group(1)
                value = -int(suffix[1:]) if suffix.startswith("m") else int(suffix)
                resolved.append(str(value))
            else:
                resolved.append(a)
        steps.append(f"Step {i + 1}: {_OP_WORDS[op]} {' and '.join(resolved)}.")
    return steps


def _format_answer_with_steps(steps, final_answer) -> str:
    if not steps:
        return str(final_answer)
    return "\n".join(steps) + f"\nAnswer: {final_answer}"


# -- Per-source normalizers --------------------------------------------

def load_financial_qa_10k() -> list:
    from datasets import load_dataset
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    first = ds[0]
    assert {"question", "answer", "context"} <= set(first.keys()), (
        f"financial-qa-10K schema mismatch, got fields: {list(first.keys())}"
    )
    out = []
    for row in ds:
        user = (
            f"Context (from {row.get('ticker', 'a company')}'s "
            f"{row.get('filing', '10-K')} filing):\n{row['context']}\n\n"
            f"Question: {row['question']}"
        )
        out.append(_make_example(user, row["answer"], "financial-qa-10K",
                                 f"10K:{_text_key(row['context'])}"))
    return out


def load_convfinqa() -> list:
    from datasets import load_dataset
    ds = load_dataset("FinGPT/fingpt-convfinqa", split="train")
    first = ds[0]
    assert {"instruction", "input", "output"} <= set(first.keys()), (
        f"fingpt-convfinqa schema mismatch, got fields: {list(first.keys())}"
    )
    out = []
    for row in ds:
        user = f"{row['instruction']}\n\n{row['input']}"
        # One row per conversation turn; later turns repeat the same
        # document, so the document text (before the Q/A history) is the group.
        doc = row["input"].split("\nQuestion:")[0]
        # Output stays a bare number, as-is -- see module docstring.
        out.append(_make_example(user, str(row["output"]), "ConvFinQA",
                                 f"ConvFinQA:{_text_key(doc)}"))
    return out


def _finqa_text_field(value) -> str:
    """pre_text/post_text in wandb/finqa-data-processed are stored as
    the *string repr* of a Python list (confirmed via a real loaded
    row: type is str, value looks like "['sentence one .', 'sentence
    two .']"), not an actual list -- ast.literal_eval recovers the
    real list; falls back to the raw string if that ever doesn't hold
    (e.g. a genuinely plain-string row), rather than crash.
    """
    if isinstance(value, list):
        return " ".join(value)
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return " ".join(parsed)
    except (ValueError, SyntaxError):
        pass
    return value


def load_finqa() -> list:
    from datasets import load_dataset
    # dreamerdeo/finqa uses a legacy loading script the current
    # `datasets` library refuses to run at all (confirmed: raises
    # "Dataset scripts are no longer supported"). wandb/finqa-data-
    # processed loads cleanly and keeps the program field the
    # dreamerdeo mirror is missing -- checked live, not assumed.
    ds = load_dataset("wandb/finqa-data-processed", split="train")
    first = ds[0]
    assert {"pre_text", "post_text", "table", "query", "output", "program", "id"} <= set(first.keys()), (
        f"FinQA schema mismatch, got fields: {list(first.keys())}"
    )
    fallback_count = 0
    out = []
    for row in ds:
        pre = _finqa_text_field(row["pre_text"])
        post = _finqa_text_field(row["post_text"])
        # table here is already rendered as a plain-text grid by this
        # mirror (confirmed live: "Row 1: ..., Row 2: ..." dash-
        # delimited text, not a list of lists) -- used as-is, not
        # re-rendered through render_table() (which assumes a real
        # list-of-lists and would iterate this string's characters).
        table_text = row["table"]
        user = f"{pre}\n\n{table_text}\n\n{post}\n\nQuestion: {row['query']}"
        answer = row["output"]
        steps = _program_to_steps(row.get("program"))
        if steps is None:
            fallback_count += 1
        target = _format_answer_with_steps(steps, answer)
        # id is "<page>-<question number>", e.g. "ABMD/2015/page_53.pdf-2".
        out.append(_make_example(user, target, "FinQA",
                                 f"FinQA:{row['id'].rsplit('-', 1)[0]}"))
    if fallback_count:
        print(f"FinQA: {fallback_count}/{len(out)} rows fell back to plain "
              f"question->answer (program not present or unparseable)")
    return out


def load_tatqa() -> list:
    from datasets import load_dataset
    ds = load_dataset("next-tat/TAT-QA", split="train")
    first = ds[0]
    assert {"table", "paragraphs", "questions"} <= set(first.keys()), (
        f"TAT-QA schema mismatch, got fields: {list(first.keys())}"
    )
    out = []
    for context_idx, context_row in enumerate(ds):
        table_text = render_table(context_row["table"].get("table", context_row["table"]))
        paragraphs = {p["order"]: p["text"] for p in context_row["paragraphs"]}
        for q in context_row["questions"]:
            rel = q.get("rel_paragraphs") or []
            rel_text = "\n".join(paragraphs[o] for o in rel if o in paragraphs)
            user = (
                f"Table:\n{table_text}\n\n"
                f"{rel_text}\n\nQuestion: {q['question']}"
            )
            answer = q["answer"]
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            derivation = q.get("derivation")
            scale = q.get("scale")
            target = answer if not scale or scale == "None" else f"{answer} {scale}"
            if derivation:
                target = f"Step 1: compute {derivation}.\nAnswer: {target}"
            out.append(_make_example(user, target, "TAT-QA", f"TAT-QA:{context_idx}"))
    return out


# -- Combine, dedup, split, audit ---------------------------------------

def _question_hash(example: dict) -> str:
    q = example["messages"][1]["content"]
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()


def dedup(examples: list) -> list:
    seen = set()
    out = []
    dropped = 0
    for ex in examples:
        h = _question_hash(ex)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(ex)
    if dropped:
        print(f"Dedup: dropped {dropped} near-duplicate examples "
              f"(hash collision on the user-message text)")
    return out


def _long_sentences(text: str) -> set:
    """Fingerprint of a document: its long sentences, normalized to
    lowercase letters and digits only. FinQA and ConvFinQA both store the
    same S&P 500 report text as " . "-separated sentences, just wrapped
    differently -- normalizing makes a shared sentence compare equal.
    Short sentences (< 60 chars) are skipped: boilerplate like "in
    millions" would link unrelated documents together.
    """
    out = set()
    for sentence in re.split(r"\s\.\s|\n", text):
        normalized = re.sub(r"[^a-z0-9]+", "", sentence.lower())
        if len(normalized) >= 60:
            out.add(normalized)
    return out


def link_overlapping_documents(examples: list) -> list:
    """ConvFinQA was built from FinQA's own source reports, so the same
    document shows up in both under different group keys. Any ConvFinQA
    document sharing >= 2 long sentences with a FinQA page gets merged
    into that page's group (union-find, so chains of matches merge too).
    """
    index = {}  # long sentence -> FinQA groups containing it
    for ex in examples:
        if ex["source"] == "FinQA":
            for s in _long_sentences(ex["messages"][1]["content"]):
                index.setdefault(s, set()).add(ex["group"])

    parent = {}

    def find(g):
        while parent.get(g, g) != g:
            g = parent[g]
        return g

    seen, linked = set(), 0
    for ex in examples:
        if ex["source"] != "ConvFinQA" or ex["group"] in seen:
            continue
        seen.add(ex["group"])
        hits = Counter(g for s in _long_sentences(ex["messages"][1]["content"])
                       for g in index.get(s, ()))
        matched = [g for g, n in hits.items() if n >= 2]
        for g in matched:
            parent[find(g)] = find(ex["group"])
        linked += bool(matched)

    for ex in examples:
        ex["group"] = find(ex["group"])
    print(f"Linked {linked}/{len(seen)} ConvFinQA documents to FinQA pages "
          f"from the same report")
    return examples


def split(examples: list, seed: int = SEED) -> tuple:
    # Split by document group, not by example: every question about one
    # document lands in the same split, so test never shares a document
    # (or an earlier ConvFinQA turn's answer) with train.
    rng = random.Random(seed)
    by_group = {}
    for ex in examples:
        by_group.setdefault(ex["group"], []).append(ex)
    groups = sorted(by_group)
    rng.shuffle(groups)
    target = int(len(examples) * 0.05)
    train, val, test = [], [], []
    for g in groups:
        bucket = val if len(val) < target else test if len(test) < target else train
        bucket.extend(by_group[g])
    rng.shuffle(train)  # groups were added whole; mix them for training order
    seen = [{ex["group"] for ex in part} for part in (train, val, test)]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]), (
        "a document group landed in more than one split"
    )
    return train, val, test


def audit_lengths(examples: list, tokenizer_name: str, max_seq_len: int) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = []
    for ex in examples:
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        lengths.append(len(tok(text)["input_ids"]))
    lengths.sort()
    n = len(lengths)
    truncated = sum(1 for l in lengths if l > max_seq_len)
    print(f"Token length audit ({n} examples): "
          f"p50={lengths[n // 2]} p90={lengths[int(n * 0.9)]} "
          f"p99={lengths[int(n * 0.99)]} max={lengths[-1]}")
    print(f"At max_seq_len={max_seq_len}: {truncated}/{n} "
          f"({100 * truncated / n:.1f}%) examples would be truncated")


def write_jsonl(examples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="training/qlora_financial/data")
    ap.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    loaders = {
        "financial-qa-10K": load_financial_qa_10k,
        "ConvFinQA": load_convfinqa,
        "FinQA": load_finqa,
        "TAT-QA": load_tatqa,
    }
    all_examples = []
    for name, loader in loaders.items():
        rows = loader()
        print(f"{name}: {len(rows)} examples")
        all_examples.extend(rows)
    print(f"Combined: {len(all_examples)} examples before dedup")

    all_examples = dedup(all_examples)
    print(f"Combined: {len(all_examples)} examples after dedup")
    all_examples = link_overlapping_documents(all_examples)

    train, val, test = split(all_examples, seed=args.seed)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    audit_lengths(all_examples, args.tokenizer, args.max_seq_len)

    out_dir = Path(args.out_dir)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    print(f"Wrote train/val/test JSONL to {out_dir}")


if __name__ == "__main__":
    main()
