"""Synthetic served traffic with a known change, to test the drift check (drift.py): windows
of test questions (never trained on), either a plain random mix or deliberately shifted.
Each query keeps its source label only so tests can check the mix; drift.py never reads it."""
import argparse
import json
import random

from training.qlora_financial.drift import features


def window(rows: list, kind: str, n: int = 500, seed: int = 0, mix: float = 1.0) -> list:
    """n queries [{"prompt", "source"}]. kind: "none" (random), "source" (80% TAT-QA) or
    "length" (only the longest third of documents). mix = the share of the window that is
    shifted; the rest is random traffic, so mix 0..1 gives a gradual change."""
    rng = random.Random(seed)
    k = 0 if kind == "none" else round(n * mix)
    if kind == "source":
        tat, other = [r for r in rows if r["source"] == "TAT-QA"], [r for r in rows if r["source"] != "TAT-QA"]
        picked = rng.sample(tat, round(k * 0.8)) + rng.sample(other, k - round(k * 0.8))
    elif kind == "length":
        doc_words = lambda r: features(r["messages"][1]["content"])["document_words"]
        picked = rng.sample(sorted(rows, key=doc_words, reverse=True)[:len(rows) // 3], k)
    elif kind == "none":
        picked = []
    else:
        raise ValueError(f"unknown kind {kind!r}: none, source or length")
    taken = {id(r) for r in picked}
    picked += rng.sample([r for r in rows if id(r) not in taken], n - k)
    rng.shuffle(picked)
    return [{"prompt": r["messages"][1]["content"], "source": r["source"]} for r in picked]


def main():
    ap = argparse.ArgumentParser(description="Write a synthetic query window (queries.jsonl).")
    ap.add_argument("--kind", default="none", choices=["none", "source", "length"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=float, default=1.0, help="share of the window that is shifted")
    ap.add_argument("--test", default="training/qlora_financial/data/v3/test.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.test) as f:
        rows = [json.loads(line) for line in f]
    with open(args.out, "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in window(rows, args.kind, args.n, args.seed, args.mix))
    print(f"{args.n} queries ({args.kind}, mix {args.mix}, seed {args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
