"""The retraining pipeline's logic (the Airflow DAG in mlops/dags/ only calls these): fine-tune
data that looks like the drifted traffic, scoring the fine-tuned candidate, the gate."""
import bisect
import json
import random
import shutil
import statistics
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.qlora_financial.canary import pick
from training.qlora_financial.drift import features
from training.qlora_financial.promote import decide
from training.qlora_financial.sample_eval import chat, load_questions


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


def score(url: str, model: str, data_dir, n: int = 300, seed: int = 0, workers: int = 8, timeout: float = 120) -> list:
    """The served candidate's greedy answers to a fixed random n test questions (picked as the
    canary picks), as rows the gate reads. n, not all 1,404: the GPU session stays under an hour.
    Any unanswered question raises (the task fails and retries): a gate on part of them would
    compare unlike with like."""
    questions = pick(load_questions(str(data_dir)), n, seed)
    def ask(q):
        answer = chat(url, q["messages"], 1, 0.0, model=model, timeout=timeout)[0]
        if not isinstance(answer, str):
            raise ValueError(f"question {q['id']}: the answer isn't text")
        return answer
    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, questions))
    return [{"id": q["id"], "source": q["source"], "gold": q["gold"], "greedy": a, "samples": []}
            for q, a in zip(questions, answers)]


def gate(candidate_rows: list, champion_rows: list) -> dict:
    """promote.decide on the candidate's questions only: the champion's saved answers cover the
    whole test set, the candidate's only a sample. Refuses saved answers to other questions (ids
    are line numbers, so another test file would match by id but not by question)."""
    saved = {r["id"]: r for r in champion_rows}
    if missing := [r["id"] for r in candidate_rows if r["id"] not in saved]:
        raise ValueError(f"the champion has no saved answer to {len(missing)} of the candidate's questions")
    if other := sum(saved[r["id"]]["gold"] != r["gold"] for r in candidate_rows):
        raise ValueError(f"saved answers are for other questions: right answer differs on {other}")
    return decide(candidate_rows, [saved[r["id"]] for r in candidate_rows])


def loaded(url: str, timeout: float = 30) -> set:
    """The names vLLM answers to: the base model and every loaded adapter."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=timeout) as resp:
        return {m["id"] for m in json.load(resp)["data"]}


def _post(url: str, body: dict, timeout: float) -> None:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=timeout).close()  # a refusal (4xx/5xx) raises HTTPError


def load_adapter(url: str, name: str, path, timeout: float = 300) -> None:
    """Load an adapter into the running vLLM under `name` (needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=True;
    `path` as vLLM sees it). Already loaded = nothing to do, so a retried task doesn't fail on
    vLLM's "already loaded" refusal."""
    if name not in loaded(url):
        _post(f"{url}/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}, timeout)


def unload_adapter(url: str, name: str, timeout: float = 60) -> None:
    _post(f"{url}/v1/unload_lora_adapter", {"lora_name": name}, timeout)
