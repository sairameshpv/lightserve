"""The retraining pipeline's logic (the Airflow DAG in mlops/dags/ only calls these): fine-tune
data that looks like the drifted traffic."""
import bisect
import random
import statistics
from collections import Counter

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
