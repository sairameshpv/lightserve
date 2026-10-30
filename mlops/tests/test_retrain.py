"""retrain.py: building fine-tune data that looks like the drifted traffic, scoring a candidate
and the gate. Made-up rows only (CI has no copy of the training data); a fake chat server."""
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlops.retrain import build_data, gate, load_adapter, loaded, score, select_like, unload_adapter
from training.qlora_financial.canary import pick
from training.qlora_financial.sample_eval import load_questions


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


@pytest.fixture
def server():
    """A fake vLLM: chat replies server.reply(prompt) (None = a server error) and records the settings
    asked with; adapters load/unload by name (server.refuse = load fails, as for a bad path)."""
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):  # /v1/models: the base model and every loaded adapter
            self.send(200, {"data": [{"id": m} for m in ["base", *srv.adapters]]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/load_lora_adapter":
                if srv.refuse or body["lora_name"] in srv.adapters:  # vLLM refuses a name twice
                    return self.send(400, {"error": "refused"})
                srv.adapters[body["lora_name"]] = body["lora_path"]
                srv.loads += 1
                return self.send(200, {})
            if self.path == "/v1/unload_lora_adapter":
                return self.send(200, {}) if srv.adapters.pop(body["lora_name"], None) else self.send(404, {})
            srv.asked.add((body["model"], body["n"], body["temperature"], body["max_tokens"]))
            answer = srv.reply(body["messages"][-1]["content"])
            self.send_response(500 if answer is None else 200)
            self.end_headers()
            if answer is not None:
                self.wfile.write(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 64

    srv = Server(("127.0.0.1", 0), Handler)
    srv.asked, srv.url = set(), f"http://127.0.0.1:{srv.server_address[1]}"
    srv.adapters, srv.loads, srv.refuse = {"v3": "/adapters/v3"}, 0, False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def questions_file(tmp_path, n=400):
    """test.jsonl with n questions: prompt "q<i>", right answer "Answer: <i>"."""
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(
        {"source": "FinQA" if i % 2 else "TAT-QA", "messages": [{"role": "system", "content": ""},
         {"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"Answer: {i}"}]}) + "\n"
        for i in range(n)))
    return tmp_path


def test_score_asks_the_same_300_once_greedy_and_returns_rows_the_gate_reads(server, tmp_path):
    server.reply = lambda prompt: "Answer: " + prompt[1:]
    rows = score(server.url, "candidate", questions_file(tmp_path), n=300)
    assert [r["id"] for r in rows] == [q["id"] for q in pick(load_questions(tmp_path), n=300)]
    assert rows[0] == {"id": rows[0]["id"], "source": rows[0]["source"], "gold": f"Answer: {rows[0]['id']}",
                       "greedy": f"Answer: {rows[0]['id']}", "samples": []}
    assert server.asked == {("candidate", 1, 0.0, 256)}  # greedy, one answer: as the champion's saved ones


def test_score_fails_if_any_question_is_unanswered(server, tmp_path):
    server.reply = lambda prompt: None if prompt == "q7" else "Answer: " + prompt[1:]
    with pytest.raises(Exception):
        score(server.url, "candidate", questions_file(tmp_path, n=10), n=10)


def answers(ids, wrong=()):
    return [{"id": i, "source": "FinQA" if i % 2 else "TAT-QA", "gold": f"Answer: {i}",
             "greedy": "Answer: -1" if i in wrong else f"Answer: {i}", "samples": []} for i in ids]


def test_gate_compares_with_the_champion_on_the_candidates_questions_only():
    champion = answers(range(400), wrong=range(0, 400, 7))  # the full test set; wrong on 58
    better = answers(range(100, 400))  # 300 of them, all right
    d = gate(better, champion)
    assert d["passed"] and d["only_candidate"] == 43 and d["only_champion"] == 0
    worse = answers(range(100, 400), wrong=range(101, 400, 5))
    d = gate(worse, champion)
    assert not d["passed"] and d["reasons"]


def test_gate_refuses_champion_answers_to_other_questions():
    champion = answers(range(400))
    champion[150]["gold"] = "Answer: 42"  # same id, another question (ids are line numbers)
    with pytest.raises(ValueError, match="other questions"):
        gate(answers(range(100, 400)), champion)
    with pytest.raises(ValueError, match="no saved answer"):
        gate(answers(range(300, 500)), answers(range(400)))


def test_load_adapter_once_even_when_retried(server):
    assert loaded(server.url) == {"base", "v3"}
    load_adapter(server.url, "run1", "/runs/run1/adapter")
    load_adapter(server.url, "run1", "/runs/run1/adapter")  # an Airflow retry: already there, no second load
    assert server.adapters["run1"] == "/runs/run1/adapter" and server.loads == 1
    assert loaded(server.url) == {"base", "v3", "run1"}


def test_unload_adapter_removes_it(server):
    load_adapter(server.url, "run1", "/runs/run1/adapter")
    unload_adapter(server.url, "run1")
    assert loaded(server.url) == {"base", "v3"}


def test_a_refused_load_raises(server):
    server.refuse = True
    with pytest.raises(Exception):
        load_adapter(server.url, "run1", "/no/such/adapter")
