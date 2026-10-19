"""DPO pairs for the PPO / GRPO / DPO comparison: v3 writes 4 answers per
question (temperature 1.0, as in the rollouts), the RL reward scores them, and
each question with both a right and a wrong answer gives one chosen/rejected
pair in OpenRLHF's DPO format.

Run (with v3 merged served by vLLM as "v2", like sample_eval):
    python3 -m training.qlora_financial.rl_pairs --url http://localhost:8000 [--limit 20]
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.qlora_financial.rl_data import stratified_sample
from training.qlora_financial.rl_reward import score
from training.qlora_financial.sample_eval import chat


def build_pairs(rows: list, seed: int = 42) -> tuple:
    """rows: {prompt, label, source, group, answers}. Returns (pairs, counts)."""
    rng, pairs, counts = random.Random(seed), [], Counter()
    for r in rows:
        right = [a for a in r["answers"] if score(a, r["label"]) == 1.0]
        wrong = [a for a in r["answers"] if score(a, r["label"]) == 0.0]
        counts["all_right" if not wrong else "all_wrong" if not right else "mixed"] += 1
        if right and wrong:
            pairs.append({"prompt": r["prompt"], "chosen": [{"role": "assistant", "content": rng.choice(right)}],
                          "rejected": [{"role": "assistant", "content": rng.choice(wrong)}],
                          "source": r["source"], "group": r["group"]})
            counts[f"pairs_{r['source']}"] += 1
    return pairs, counts


def generate(url: str, rows: list, out_path: Path, n: int = 4, concurrency: int = 32) -> list:
    """Add n answers to each row (keyed by its line number in the question
    file), appending as they finish; rerun to resume after a crash."""
    done = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(out_path)} if out_path.exists() else {}
    with open(out_path, "a") as f, ThreadPoolExecutor(concurrency) as pool:
        futs = {pool.submit(chat, url, r["prompt"], n, 1.0, 512): i for i, r in enumerate(rows) if i not in done}
        for k, fut in enumerate(as_completed(futs), 1):
            done[futs[fut]] = fut.result()
            f.write(json.dumps({"id": futs[fut], "answers": done[futs[fut]]}) + "\n")
            f.flush()
            if k % 200 == 0 or k == len(futs):
                print(f"{k}/{len(futs)} questions")
    return [{**r, "answers": done[i]} for i, r in enumerate(rows)]


def subset(data: Path, samples: Path, n: int, tag: str) -> Counter:
    """Seeded, source-stratified n of train.jsonl -> train_{tag}.jsonl, and
    dpo_train_{tag}.jsonl from the answers already saved for those questions
    (no new generation, so DPO's budget is the same 4 answers per question)."""
    rows = [{**json.loads(l), "id": i} for i, l in enumerate(open(data / "train.jsonl"))]
    saved = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(samples / "samples_train.jsonl")}
    picked = stratified_sample(rows, n)
    pairs, counts = build_pairs([{**r, "answers": saved[r["id"]]} for r in picked])
    for name, out in ((f"train_{tag}", picked), (f"dpo_train_{tag}", pairs)):
        with open(data / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
    counts["questions"] = len(picked)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--data-dir", default="training/qlora_financial/data/rl")
    ap.add_argument("--samples-dir", default="training/qlora_financial/outputs/rl/pairs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None, help="only write a stratified train subset of this size")
    args = ap.parse_args()
    data, samples = Path(args.data_dir), Path(args.samples_dir)
    if args.subset:
        tag = f"{args.subset // 1000}k" if args.subset % 1000 == 0 else str(args.subset)
        print(dict(sorted(subset(data, samples, args.subset, tag).items())))
        return
    samples.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(data / f"{split}.jsonl")][: args.limit]
        pairs, counts = build_pairs(generate(args.url, rows, samples / f"samples_{split}.jsonl"))
        with open(data / f"dpo_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in pairs)
        print(split, dict(sorted(counts.items())), f"-> {len(pairs)} pairs")


if __name__ == "__main__":
    main()
