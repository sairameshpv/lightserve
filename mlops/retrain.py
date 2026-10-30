"""The retraining pipeline's logic (the Airflow DAG in mlops/dags/ only calls these): fine-tune
data that looks like the drifted traffic."""
import bisect
import json
import random
import shutil
import statistics
from collections import Counter
from pathlib import Path

from training.qlora_financial.drift import features


def select_like(train_rows: list, window_prompts: list, drifted_columns: list, n: int, seed: int = 0) -> list:
    """Up to n training rows of the kinds the query window has MORE of than training. Each row
    falls into a band (fifth of training's spread) per changed measurement; a combination of
    bands that makes up a bigger share of the window than of training gets picked, the more the
    bigger the excess (window share / training share - 1). Kinds the window has less of are never
    picked: the replay part of the fine-tune data covers them."""
    if not drifted_columns:
        raise ValueError("no measurement changed: nothing to look like")
    train_feats = [features(r["messages"][1]["content"]) for r in train_rows]
    cuts = {c: statistics.quantiles([f[c] for f in train_feats], n=5, method="inclusive") for c in drifted_columns}

    def band(f):
        return tuple(bisect.bisect_right(cuts[c], f[c]) for c in drifted_columns)

    train_bands = [band(f) for f in train_feats]
    t_share = {b: k / len(train_rows) for b, k in Counter(train_bands).items()}
    w_share = {b: k / len(window_prompts) for b, k in Counter(band(features(p)) for p in window_prompts).items()}
    weight = {b: w_share.get(b, 0.0) / s - 1 for b, s in t_share.items()}
    rng = random.Random(seed)  # weighted picking without repeats: largest u ** (1 / weight) wins
    keyed = [(rng.random() ** (1 / weight[b]), i) for i, b in enumerate(train_bands) if weight[b] > 0]
    return [train_rows[i] for _, i in sorted(keyed, reverse=True)[:n]]


def build_data(train_dir, window_prompts: list, drifted_columns: list, out_dir, n_like: int, seed: int = 0) -> dict:
    """The fine-tune data in out_dir: train.jsonl = up to n_like rows like the traffic
    (select_like) + as many replayed from the rest at random, so the model doesn't forget the
    other kinds; shuffled; rows copied unchanged. val/test copied as they are, so scoring and
    lineage see the same questions as the champion's."""
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    lines = (train_dir / "train.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    line_of = {id(r): line for r, line in zip(rows, lines)}
    like = select_like(rows, window_prompts, drifted_columns, n_like, seed)
    taken = {id(r) for r in like}
    rng = random.Random(seed)
    replay = rng.sample([r for r in rows if id(r) not in taken], len(like))
    mixed = like + replay
    rng.shuffle(mixed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("".join(line_of[id(r)] + "\n" for r in mixed))
    for split in ("val", "test"):
        shutil.copyfile(train_dir / f"{split}.jsonl", out_dir / f"{split}.jsonl")
    return {"like": len(like), "replay": len(replay), "train": len(mixed)}
