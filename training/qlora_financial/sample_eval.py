"""How much room does v2 have to improve? Greedy + 16 sampled answers per
numeric test question, served by vLLM with the v2 LoRA adapter, scored by
verify_finetune.numbers_match. Reports greedy accuracy, pass@8, maj@8 and
always/sometimes/never-right buckets.
"""

import argparse
import json
import math
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.qlora_financial.verify_finetune import SEED, _SINGLE_NUMBER_RE, _final_answer, numbers_match, parse_number


def load_questions(data_dir: str) -> list:
    """Every test row whose correct final answer is a single number (same
    rule as verify_finetune.pick_examples), in file order. id = line
    number in test.jsonl, so results can be matched back to rows."""
    rows = [json.loads(line) for line in open(Path(data_dir) / "test.jsonl")]
    return [{"id": i, "source": r["source"], "messages": r["messages"][:2],
             "gold": r["messages"][2]["content"], "gold_num": parse_number(r["messages"][2]["content"])}
            for i, r in enumerate(rows)
            if _SINGLE_NUMBER_RE.fullmatch(_final_answer(r["messages"][2]["content"]))]


def chat(url: str, messages: list, n: int, temperature: float, max_tokens: int = 256) -> list:
    """n answers from vLLM's OpenAI-style chat endpoint (served model "v2"
    = base + adapter). max_tokens matches verify_finetune.generate."""
    body = {"model": "v2", "messages": messages, "n": n, "temperature": temperature,
            "top_p": 1.0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return [c["message"]["content"] for c in json.load(resp)["choices"]]


def answer_question(url: str, q: dict, n_samples: int, temperature: float) -> dict:
    """One greedy answer + n_samples sampled answers, each parsed and scored."""
    answers = chat(url, q["messages"], 1, 0.0) + chat(url, q["messages"], n_samples, temperature)
    nums = [parse_number(a) for a in answers]
    ok = [numbers_match(x, q["gold_num"]) for x in nums]
    return {"id": q["id"], "source": q["source"], "gold": q["gold"], "gold_num": q["gold_num"],
            "greedy": answers[0], "greedy_num": nums[0], "greedy_ok": ok[0],
            "samples": answers[1:], "sample_nums": nums[1:], "sample_ok": ok[1:]}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: the chance that at
    least one of k samples drawn without replacement is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def majority_vote(nums: list):
    """Most common parsed number, grouping values within 1% (as
    numbers_match does). None answers never win. Ties: first seen."""
    groups = []  # [representative value, count]
    for x in nums:
        if x is None:
            continue
        for g in groups:
            if math.isclose(x, g[0], rel_tol=0.01, abs_tol=1e-6):
                g[1] += 1
                break
        else:
            groups.append([x, 1])
    return max(groups, key=lambda g: g[1])[0] if groups else None


def bucket(c: int, n: int) -> str:
    return "never" if c == 0 else "always" if c == n else "sometimes"


def run_all(url: str, questions: list, out_path: Path, n_samples: int, temperature: float, concurrency: int):
    """Answer every question not already in out_path, `concurrency` at a
    time, appending each finished row right away (so a crash or
    preemption loses nothing: rerun and it resumes)."""
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go")
    t0 = time.time()
    with open(out_path, "a") as f, ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(answer_question, url, q, n_samples, temperature) for q in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done, {time.time() - t0:.0f} s")


def maj_at_k(row: dict, k: int, rng: random.Random, trials: int = 100) -> float:
    """Share of `trials` random k-of-n subsets of the samples whose
    majority answer is correct (averages out which k we happened to pick)."""
    picks = (majority_vote(rng.sample(row["sample_nums"], k)) for _ in range(trials))
    return sum(numbers_match(p, row["gold_num"]) for p in picks) / trials


def report(rows: list, k: int = 8) -> dict:
    """Per-source + ALL table: greedy, mean sampled, pass@k, maj@k, maj@n
    accuracy and always/sometimes/never-right shares."""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
        groups["ALL"].append(r)
    summary = {}
    print(f"\n{'dataset':18} {'n':>5} {'greedy':>7} {'sampled':>8} {f'pass@{k}':>7} {f'maj@{k}':>7} "
          f"{'maj@all':>8} {'always':>7} {'sometimes':>10} {'never':>6}")
    for src in sorted(groups, key=lambda s: (s == "ALL", s)):  # ALL last
        g, mean = groups[src], lambda xs: sum(xs) / len(xs)
        n = len(g[0]["sample_ok"])
        s = {"n": len(g), "greedy": mean([r["greedy_ok"] for r in g]),
             "sampled": mean([mean(r["sample_ok"]) for r in g]),
             f"pass@{k}": mean([pass_at_k(n, sum(r["sample_ok"]), k) for r in g]),
             f"maj@{k}": mean([maj_at_k(r, k, rng) for r in g]),
             "maj@all": mean([numbers_match(majority_vote(r["sample_nums"]), r["gold_num"]) for r in g])}
        for b in ("always", "sometimes", "never"):
            s[b] = mean([bucket(sum(r["sample_ok"]), n) == b for r in g])
        summary[src] = s
        print(f"{src:18} {s['n']:5} " + " ".join(f"{s[c]:{w}.1%}" for c, w in zip(list(s)[1:], (7, 8, 7, 7, 8, 7, 10, 6))))
    return summary


def compare_old(rows: list, questions: list, old_path: Path) -> dict:
    """Setup check: vLLM greedy vs verify_finetune's HF greedy ("tuned")
    on the questions both scored, matched by question text + gold."""
    key_to_id = {(q["messages"][1]["content"], q["gold"]): q["id"] for q in questions}
    old = {key_to_id.get((o["question"], o["gold"])): o["tuned_ok"]
           for o in map(json.loads, open(old_path)) if o["kind"] == "scored"}
    pairs = [(old[r["id"]], r["greedy_ok"]) for r in rows if r["id"] in old]
    if not pairs:
        return {}
    c = {"n": len(pairs), "old_acc": sum(o for o, _ in pairs) / len(pairs),
         "vllm_acc": sum(v for _, v in pairs) / len(pairs),
         "old_only": sum(o and not v for o, v in pairs), "vllm_only": sum(v and not o for o, v in pairs)}
    print(f"\nSetup check on {c['n']} questions also scored by verify_finetune: old (HF) {c['old_acc']:.1%}, "
          f"vLLM greedy {c['vllm_acc']:.1%}; right only in old {c['old_only']}, only in vLLM {c['vllm_only']}")
    return c


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--data-dir", default="training/qlora_financial/data")
    ap.add_argument("--out", default="training/qlora_financial/outputs/v2/samples.jsonl")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, help="only the first N questions (trial run)")
    ap.add_argument("--compare", default="training/qlora_financial/outputs/v2/verify_results.jsonl",
                    help="verify_finetune results to check the vLLM setup against (skipped if missing)")
    return ap.parse_args()


def main():
    args = parse_args()
    questions = load_questions(args.data_dir)[:args.limit]
    out = Path(args.out)
    run_all(args.url, questions, out, args.n_samples, args.temperature, args.concurrency)
    wanted = {q["id"] for q in questions}
    rows = [r for r in map(json.loads, open(out)) if r["id"] in wanted]
    summary = report(rows)
    old_path = Path(args.compare)
    check = compare_old(rows, questions, old_path) if old_path.exists() else {}
    summary_path = out.with_name("sample_eval_summary.json")
    summary_path.write_text(json.dumps({"args": vars(args), "results": summary, "setup_check": check}, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
