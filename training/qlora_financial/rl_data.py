"""Question sets for the PPO / GRPO / DPO comparison: numeric questions from
the v3 data (prepare_dataset --convfinqa-steps), as OpenRLHF prompt/label
rows. Train and val only -- test is never touched -- with a fixed seed and
the same share of each source as the full split.

Run:
    python3 -m training.qlora_financial.rl_data
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from training.qlora_financial.verify_finetune import SEED, _SINGLE_NUMBER_RE, _final_answer


def numeric_rows(path: Path) -> list:
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if _SINGLE_NUMBER_RE.fullmatch(_final_answer(r["messages"][2]["content"]))]


def stratified_sample(rows: list, n: int, seed: int = SEED) -> list:
    """n rows with each source's share as in `rows` (rounded), seeded."""
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    rng = random.Random(seed)
    return [r for src in sorted(by_source)
            for r in rng.sample(by_source[src], round(n * len(by_source[src]) / len(rows)))]


def to_openrlhf(r: dict) -> dict:
    """`prompt` = system + user messages (chat-templated by OpenRLHF), `label` = the correct answer text."""
    return {"prompt": r["messages"][:2], "label": r["messages"][2]["content"], "source": r["source"], "group": r["group"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="training/qlora_financial/data/v3")
    ap.add_argument("--out-dir", default="training/qlora_financial/data/rl")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=300)
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out_dir)
    test_groups = {json.loads(line)["group"] for line in open(data / "test.jsonl")}
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        rows = stratified_sample(numeric_rows(data / f"{split}.jsonl"), n)
        assert not {r["group"] for r in rows} & test_groups, f"{split} shares a document with test"
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(to_openrlhf(r)) + "\n")
        print(f"{split}: {len(rows)} rows -> {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
