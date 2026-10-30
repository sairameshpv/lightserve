"""retrain.py: building fine-tune data that looks like the drifted traffic. Made-up rows only
(CI has no copy of the training data)."""
import json
from collections import Counter

import pytest

from mlops.retrain import build_data, select_like


def row(i, kind):
    """kind "table": a short document with a tab table and many digits (like TAT-QA);
    "text": a long plain document (like FinQA)."""
    if kind == "table":
        doc = "\n".join(f"Revenue\t{100 + i}\t{200 + i}" for _ in range(4))
    else:
        doc = " ".join(["the company reported results for the year"] * (40 + i % 7))
    prompt = f"{doc}\nQuestion: what was it in {2000 + i}?"
    return {"source": kind, "messages": [{"role": "system", "content": ""}, {"role": "user", "content": prompt},
                                         {"role": "assistant", "content": "Answer: 1"}]}


def test_select_like_picks_rows_that_look_like_the_drifted_window():
    train = [row(i, "table") for i in range(35)] + [row(i, "text") for i in range(35, 100)]  # 35% tables
    window = [row(i, "table")["messages"][1]["content"] for i in range(200, 280)] + \
             [row(i, "text")["messages"][1]["content"] for i in range(300, 320)]  # 80% tables
    picked = select_like(train, window, ["document_words", "table_rows", "digit_share"], n=20, seed=0)
    assert len(picked) == 20 and len({id(r) for r in picked}) == 20
    assert Counter(r["source"] for r in picked)["table"] >= 18  # training is 35% tables
    assert picked == select_like(train, window, ["document_words", "table_rows", "digit_share"], n=20, seed=0)


def test_select_like_needs_changed_measurements():
    with pytest.raises(ValueError, match="no measurement changed"):
        select_like([row(0, "text")], ["x"], [], n=1)


def test_build_data_mixes_like_the_traffic_with_an_equal_replay(tmp_path):
    src, out = tmp_path / "data", tmp_path / "run" / "data"
    src.mkdir()
    train = [row(i, "table") for i in range(35)] + [row(i, "text") for i in range(35, 100)]
    (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    for split in ("val", "test"):
        (src / f"{split}.jsonl").write_text(json.dumps(row(7, "text")) + "\n")
    window = [row(i, "table")["messages"][1]["content"] for i in range(200, 280)] + \
             [row(i, "text")["messages"][1]["content"] for i in range(300, 320)]
    counts = build_data(src, window, ["document_words", "table_rows", "digit_share"], out, n_like=20, seed=0)
    assert counts == {"like": 20, "replay": 20, "train": 40}
    lines = (out / "train.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(set(lines)) == 40 and set(lines) <= set(json.dumps(r) for r in train)  # no repeats, unchanged rows
    assert Counter(r["source"] for r in rows)["table"] >= 20  # the like part, plus any tables the replay drew
    assert [r["source"] for r in rows[:20]] != ["table"] * 20  # shuffled, not like-then-replay
    for split in ("val", "test"):  # scoring and lineage see the same questions
        assert (out / f"{split}.jsonl").read_bytes() == (src / f"{split}.jsonl").read_bytes()
