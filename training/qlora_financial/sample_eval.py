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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.qlora_financial.calculator import apply_calculator, calculate
from training.qlora_financial.verify_finetune import (SEED, _NUMBER_RE, _SCALE, _SINGLE_NUMBER_RE, _final_answer,
                                                     numbers_match, parse_number)


def load_questions(data_dir: str) -> list:
    """Every test row whose correct final answer is a single number (same
    rule as verify_finetune.pick_examples), in file order. id = line
    number in test.jsonl, so results can be matched back to rows."""
    rows = [json.loads(line) for line in open(Path(data_dir) / "test.jsonl")]
    return [{"id": i, "source": r["source"], "messages": r["messages"][:2],
             "gold": r["messages"][2]["content"], "gold_num": parse_number(r["messages"][2]["content"])}
            for i, r in enumerate(rows)
            if _SINGLE_NUMBER_RE.fullmatch(_final_answer(r["messages"][2]["content"]))]


VARIANTS = ("official", "years", "scale", "fair")  # each adds one change to the one before


def _parts(text: str):
    """(number without its scale word, scale multiplier, decimal places) of
    the last number in the final answer, as parse_number reads it; None if
    there is none."""
    matches = list(_NUMBER_RE.finditer(_final_answer(text)))
    if not matches:
        return None
    m = matches[-1]
    num = m["num"].replace(",", "")
    value = -float(num) if m["neg"] or (m["open"] and m["close"]) else float(num)
    return value, _SCALE.get(m["unit"], 1), len(num.split(".")[1]) if "." in num else 0


def match_variant(pred_text: str, gold_text: str, variant: str) -> bool:
    """Is the answer right under this checker variant? "official" is
    numbers_match. "years": a year-like gold (whole, 1900-2100, no scale)
    needs the exact year. "scale": same digits also count when a scale word
    is added or dropped (not swapped: thousand vs million stays wrong).
    "fair": 0.1% instead of 1%, or equal once rounded to the shorter
    number's decimals (39.34 vs 39.3) -- the 0.1-1% band is mostly real
    arithmetic slips (0.7168 for 151/211 = 0.71564)."""
    if variant == "official":
        return numbers_match(parse_number(pred_text), parse_number(gold_text))
    level, pred, gold = VARIANTS.index(variant), _parts(pred_text), _parts(gold_text)
    if pred is None or gold is None:
        return False
    if gold[1] == 1 and gold[0].is_integer() and 1900 <= gold[0] <= 2100:
        return pred[0] * pred[1] == gold[0]
    pairs = [(pred[0] * pred[1], gold[0] * gold[1])]
    if level >= 2 and (pred[1] == 1) != (gold[1] == 1):  # exactly one side has a scale word
        pairs.append((pred[0], gold[0]))
    if level < 3:
        return any(math.isclose(p * f, g, rel_tol=0.01, abs_tol=1e-6) for p, g in pairs for f in (1, 100, 0.01))
    # Rounding only forgives a gold that was itself rounded: the answer, at
    # least as precise, rounds to it (0.79 for 0.8). A coarser answer (0.9 for
    # 0.93, $0 for $2.0 million) must still be within 0.1%. Compared in the
    # answer's units: gold moved into them (g / f) shifts its decimals by 2.
    shift = {1: 0, 100: 2, 0.01: -2}
    return any(math.isclose(p * f, g, rel_tol=0.001, abs_tol=1e-6)
               or (pred[2] >= (d := max(gold[2] + shift[f], 0)) and round(p, d) == round(g / f, d))
               for p, g in pairs for f in (1, 100, 0.01))


def rescore(rows: list, variant: str) -> list:
    """Copies of the saved rows with greedy_ok / sample_ok recomputed under
    `variant` (no server needed: the raw answers are in each row)."""
    return [{**r, "greedy_ok": match_variant(r["greedy"], r["gold"], variant),
             "sample_ok": [match_variant(a, r["gold"], variant) for a in r["samples"]]}
            for r in rows]


def chat(url: str, messages: list, n: int, temperature: float, max_tokens: int = 256, model: str = "v2",
         timeout: float = 600) -> list:
    """n answers from vLLM's OpenAI-style chat endpoint (served model `model`, e.g. "v2"
    = base + adapter). max_tokens matches verify_finetune.generate."""
    body = {"model": model, "messages": messages, "n": n, "temperature": temperature,
            "top_p": 1.0, "max_tokens": max_tokens}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    return sum(vote_ok(row, rng.sample(range(len(row["sample_nums"])), k)) for _ in range(trials)) / trials


def vote_ok(row: dict, idx) -> bool:
    """Is the majority answer among samples `idx` right? Judged by the
    winning sample's own sample_ok, so it follows whichever checker
    (rescore variant) produced the row."""
    nums = [row["sample_nums"][i] for i in idx]
    win = majority_vote(nums)
    return win is not None and row["sample_ok"][idx[nums.index(win)]]


def report(rows: list, k: int = 8, show: bool = True) -> dict:
    """Per-source + ALL table: greedy, mean sampled, pass@k, maj@k, maj@n
    accuracy and always/sometimes/never-right shares."""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
        groups["ALL"].append(r)
    summary = {}
    if show:
        print(f"\n{'dataset':18} {'n':>5} {'greedy':>7} {'sampled':>8} {f'pass@{k}':>7} {f'maj@{k}':>7} "
              f"{'maj@all':>8} {'always':>7} {'sometimes':>10} {'never':>6}")
    for src in sorted(groups, key=lambda s: (s == "ALL", s)):  # ALL last
        g, mean = groups[src], lambda xs: sum(xs) / len(xs)
        n = len(g[0]["sample_ok"])
        s = {"n": len(g), "greedy": mean([r["greedy_ok"] for r in g]),
             "sampled": mean([mean(r["sample_ok"]) for r in g]),
             f"pass@{k}": mean([pass_at_k(n, sum(r["sample_ok"]), k) for r in g]),
             f"maj@{k}": mean([maj_at_k(r, k, rng) for r in g]),
             "maj@all": mean([vote_ok(r, range(n)) for r in g])}
        for b in ("always", "sometimes", "never"):
            s[b] = mean([bucket(sum(r["sample_ok"]), n) == b for r in g])
        summary[src] = s
        if show:
            print(f"{src:18} {s['n']:5} " + " ".join(f"{s[c]:{w}.1%}" for c, w in zip(list(s)[1:], (7, 8, 7, 7, 8, 7, 10, 6))))
    return summary


def checker_table(rows: list) -> dict:
    """ALL-row results under each checker variant, one line per variant."""
    table = {v: report(rescore(rows, v), show=False)["ALL"] for v in VARIANTS}
    cols = list(table["official"])[1:]
    print(f"\n{'checker':10} " + " ".join(f"{c:>9}" for c in cols))
    for v, s in table.items():
        print(f"{v:10} " + " ".join(f"{s[c]:9.1%}" for c in cols))
    return table


def flip_examples(rows: list, per_step: int = 15):
    """For each variant, up to per_step random answers (greedy or sampled)
    it judges differently from the variant before it, with the reasoning
    shown, so a person can tell honest rounding from a real mistake."""
    rng = random.Random(SEED)
    for prev, cur in zip(VARIANTS, VARIANTS[1:]):
        flips = [(r, a, ok) for r in rows for a in [r["greedy"]] + r["samples"]
                 if (ok := match_variant(a, r["gold"], cur)) != match_variant(a, r["gold"], prev)]
        print(f"\n===== {prev} -> {cur}: {len(flips)} answers change =====")
        for r, a, ok in rng.sample(flips, min(per_step, len(flips))):
            print(f"[id {r['id']} {r['source']}] now {'RIGHT' if ok else 'WRONG'}\n"
                  f"  correct: {' | '.join(r['gold'].splitlines())[-150:]}\n  model:   {' | '.join(a.splitlines())[-150:]}")


def gold_replay(questions: list) -> Counter:
    """Calculator safety check: run it on the CORRECT reasoning of every
    question; it should reproduce the correct answer (fair checker). Prints
    each miss so it can be explained."""
    counts = Counter()
    for q in questions:
        fixed = apply_calculator(q["gold"])
        if calculate(q["gold"]) is None:
            counts[q["source"], "not calculable"] += 1
        elif match_variant(fixed, q["gold"], "fair"):
            counts[q["source"], "reproduced"] += 1
        else:
            counts[q["source"], "MISSED"] += 1
            print(f"MISSED id {q['id']} {q['source']}: {' | '.join(q['gold'].splitlines())[-120:]}  ->  {fixed.rsplit('Answer:', 1)[1]}")
    for (src, kind), n in sorted(counts.items()):
        print(f"{src:18} {kind:15} {n}")
    return counts


def with_calculator(rows: list) -> list:
    """Copies of the rows with every answer's final number recomputed from
    its own steps (sample_nums too, so majority votes see the new numbers)."""
    return [{**r, "greedy": apply_calculator(r["greedy"]),
             "samples": (s := [apply_calculator(a) for a in r["samples"]]),
             "sample_nums": [parse_number(a) for a in s]} for r in rows]


def calculator_report(rows: list, per_step: int = 15):
    """Fair-checker results as written vs with the calculator, then why each
    wrong greedy answer (FinQA/TAT-QA) was wrong: an arithmetic slip the
    calculator fixes, a wrong method (calculable but still wrong), or no
    calculable steps. Prints per_step random wrong-method examples."""
    before, after = rescore(rows, "fair"), rescore(with_calculator(rows), "fair")
    for name, rs in (("AS WRITTEN (fair checker)", before), ("WITH CALCULATOR (fair checker)", after)):
        print(f"\n{name}", end="")
        report(rs)
    why, method = Counter(), []
    for b, a in zip(before, after):
        if b["source"] in ("FinQA", "TAT-QA") and not b["greedy_ok"]:
            kind = ("arithmetic slip (fixed)" if a["greedy_ok"] else
                    "wrong method (calculable)" if calculate(b["greedy"]) else "not calculable")
            why[b["source"], kind] += 1
            if kind.startswith("wrong method"):
                method.append(b)
    print("\nWhy greedy answers were wrong (fair checker):")
    for (src, kind), n in sorted(why.items()):
        print(f"  {src:7} {kind:28} {n}")
    print(f"\n===== {per_step} random wrong-method answers =====")
    for r in random.Random(SEED).sample(method, min(per_step, len(method))):
        print(f"[id {r['id']} {r['source']}]\n  correct: {' | '.join(r['gold'].splitlines())[-150:]}"
              f"\n  model:   {' | '.join(r['greedy'].splitlines())[-150:]}")


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
    ap.add_argument("--checkers", action="store_true",
                    help="no server: re-score saved --out answers under each checker variant + show flipped examples")
    ap.add_argument("--calculator", action="store_true",
                    help="with --checkers: calculator safety check on the correct reasoning, then v2 with vs without it")
    return ap.parse_args()


def main():
    args = parse_args()
    questions = load_questions(args.data_dir)[:args.limit]
    out = Path(args.out)
    if args.checkers:
        rows = [json.loads(line) for line in open(out)]
        if args.calculator:
            gold_replay(questions)
            calculator_report(rows)
        else:
            checker_table(rows)
            flip_examples(rows)
        return
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
