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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--data-dir", default="training/qlora_financial/data/rl")
    ap.add_argument("--samples-dir", default="training/qlora_financial/outputs/rl/pairs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    data, samples = Path(args.data_dir), Path(args.samples_dir)
    samples.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(data / f"{split}.jsonl")][: args.limit]
        pairs, counts = build_pairs(generate(args.url, rows, samples / f"samples_{split}.jsonl"))
        with open(data / f"dpo_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in pairs)
        print(split, dict(sorted(counts.items())), f"-> {len(pairs)} pairs")


if __name__ == "__main__":
    main()
