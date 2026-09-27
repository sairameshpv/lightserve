"""Original vs fine-tuned Llama-3-8B-Instruct on the held-out test split (data/test.jsonl, never seen in training).

One model load, two answers per question: the same 4-bit base as train.py, with the LoRA adapter on (fine-tuned) and switched off via
PeftModel.disable_adapter() (original) -- so the only difference between the two answers is the adapter.

Two checks:
- Readable: 2 questions per source dataset, printed side by side.
- Scored: N questions whose correct final answer is a single number,
  scored by numbers_match() for both models.

Needs an NVIDIA GPU (4-bit loading) -- run on the L40S after train.py.
Run:
    python3 -m training.qlora_financial.verify_finetune
    python3 -m training.qlora_financial.verify_finetune \\
        --adapter training/qlora_financial/outputs/smoke/adapter --n-scored 16
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="training/qlora_financial/outputs/full/adapter")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="HF model id, or a local checkpoint folder on the node")
    ap.add_argument("--data-dir", default="training/qlora_financial/data")
    ap.add_argument("--n-scored", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    return ap.parse_args()


# A correct answer counts as "a single number" if its final part is just
# a number, optionally with $, parentheses, %, or a scale word.
_SINGLE_NUMBER_RE = re.compile(r"[-$(]*[\d,]*\.?\d+%?\)?( (thousand|million|billion|percent))?")


def _final_answer(text: str) -> str:
    """The part after the last 'Answer:' (FinQA/TAT-QA put reasoning
    steps first), or the whole text if there is none."""
    return text.rsplit("Answer:", 1)[-1].strip()


def pick_examples(data_dir: str, n_scored: int, seed: int = SEED) -> tuple:
    """readable: 2 per source dataset. scored: n_scored examples whose
    correct final answer is a single number. Fixed seed, so every run
    (and both models) see the same questions."""
    rows = [json.loads(line) for line in open(Path(data_dir) / "test.jsonl")]
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    readable = [ex for src in sorted(by_source) for ex in rng.sample(by_source[src], 2)]
    numeric = [r for r in rows
               if _SINGLE_NUMBER_RE.fullmatch(_final_answer(r["messages"][2]["content"]))]
    return readable, rng.sample(numeric, min(n_scored, len(numeric)))


_NUMBER_RE = re.compile(r"(?P<open>\()?(?P<neg>-)?\$?(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)(?P<close>\))?"
                        r"\s*(?P<unit>%|percent|thousand|million|billion)?")
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def parse_number(text: str):
    """Last number in the final answer, as a float (None if there is
    none). '(35.0)' counts as negative (accounting style); scale words
    multiply ('322 million' -> 322e6); '%' is dropped ('14.1%' -> 14.1),
    numbers_match() handles percent-vs-fraction."""
    matches = list(_NUMBER_RE.finditer(_final_answer(text)))
    if not matches:
        return None
    m = matches[-1]
    value = float(m["num"].replace(",", ""))
    if m["neg"] or (m["open"] and m["close"]):
        value = -value
    return value * _SCALE.get(m["unit"], 1)


def numbers_match(pred, gold) -> bool:
    """The scoring rule: within 1% of the correct number, or of 100x /
    1/100x it (a model writing 14.1 for a correct 0.141 has the right
    number in percent form, not a wrong answer)."""
    if pred is None or gold is None:
        return False
    return any(math.isclose(p, gold, rel_tol=0.01, abs_tol=1e-6)
               for p in (pred, pred * 100, pred / 100))


def load_model(model_name: str, adapter_dir: str):
    """4-bit base with train.py's own quantization config (reused, so the
    setup matches training exactly), plus the LoRA adapter on top.
    Tokenizer pads on the left: batched generation continues each prompt
    from its end, so padding must go in front."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from training.qlora_financial.train import qlora_configs
    quant, _ = qlora_configs()
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant,
                                                dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    return model, tok


def generate(model, tok, examples: list, batch_size: int, max_new_tokens: int = 256) -> list:
    """Greedy answers (no sampling, so reruns agree) to each example's
    system+user prompt, in batches. Returns only the new text."""
    import torch
    answers = []
    for i in range(0, len(examples), batch_size):
        prompts = [tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
                   for ex in examples[i:i + batch_size]]
        # The chat template already starts with <|begin_of_text|>; don't add a second one.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        answers += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return answers


def main():
    args = parse_args()
    readable, scored = pick_examples(args.data_dir, args.n_scored)
    print(f"readable={len(readable)} scored={len(scored)}")
    model, tok = load_model(args.model, args.adapter)

    examples = readable + scored
    tuned = generate(model, tok, examples, args.batch_size)
    with model.disable_adapter():  # same weights, adapter off = the original model
        original = generate(model, tok, examples, args.batch_size)

    results = []
    for i, (ex, orig_ans, tuned_ans) in enumerate(zip(examples, original, tuned)):
        gold = ex["messages"][2]["content"]
        row = {"kind": "readable" if i < len(readable) else "scored", "source": ex["source"],
               "question": ex["messages"][1]["content"], "gold": gold,
               "original": orig_ans, "tuned": tuned_ans}
        if row["kind"] == "scored":
            gold_num = parse_number(gold)
            for who, ans in (("original", orig_ans), ("tuned", tuned_ans)):
                row[f"{who}_num"] = parse_number(ans)
                row[f"{who}_ok"] = numbers_match(row[f"{who}_num"], gold_num)
            row["gold_num"] = gold_num
        results.append(row)

    for row in results[:len(readable)]:
        print(f"\n===== {row['source']} =====")
        print("QUESTION: ..." + row["question"][-300:])  # the actual question is at the end
        print(f"CORRECT:  {row['gold']}\nORIGINAL: {row['original']}\nTUNED:    {row['tuned']}")

    groups = defaultdict(list)
    for row in results[len(readable):]:
        groups[row["source"]].append(row)
        groups["ALL"].append(row)
    accuracy = {}
    print(f"\n{'dataset':18} {'n':>4} {'original':>9} {'tuned':>9}")
    for src in sorted(groups, key=lambda s: (s == "ALL", s)):  # ALL last
        rows = groups[src]
        accuracy[src] = tuple(sum(r[f"{who}_ok"] for r in rows) / len(rows) for who in ("original", "tuned"))
        print(f"{src:18} {len(rows):4} {accuracy[src][0]:9.1%} {accuracy[src][1]:9.1%}")

    run_dir = Path(args.adapter).parent  # outputs/full or outputs/smoke
    out_path = run_dir / "verify_results.jsonl"
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(results)} rows to {out_path}")

    import os
    import mlflow
    # Same SQLite db train.py writes (outputs/mlflow.db), separate experiment.
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                           f"sqlite:///{(run_dir.parent / 'mlflow.db').resolve()}"))
    mlflow.set_experiment("qlora-financial-verify")
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({"adapter": args.adapter, "n_scored": len(scored)})
        for src, (orig_acc, tuned_acc) in accuracy.items():
            mlflow.log_metrics({f"{src}_original_acc": orig_acc, f"{src}_tuned_acc": tuned_acc})


if __name__ == "__main__":
    main()
