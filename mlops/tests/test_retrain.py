"""retrain.py: building fine-tune data that looks like the drifted traffic, scoring a candidate
and the gate. Made-up rows only (CI has no copy of the training data); a fake chat server."""
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from mlops.retrain import (add_round, build_data, canary_reading, champion_info, gate, load_adapter, loaded,
                           read_current, score,
                           select_like, set_current, unload_adapter, watch_verdict)
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


CHAMPION = {"model": "v3", "samples": "outputs/v3/samples.jsonl"}


def test_current_model_is_the_champion_until_one_is_deployed(tmp_path):
    assert read_current(tmp_path / "current.json", CHAMPION) == CHAMPION


def test_set_current_switches_and_keeps_the_one_before(tmp_path):
    path = tmp_path / "current.json"
    run1, run2 = {"model": "run1", "samples": "runs/run1/answers.jsonl"}, {"model": "run2", "samples": "x"}
    set_current(path, run1, CHAMPION)
    assert read_current(path, CHAMPION) == {**run1, "previous": CHAMPION}  # rollback target
    set_current(path, run2, CHAMPION)
    assert read_current(path, CHAMPION) == {**run2, "previous": run1}  # one step back, not a growing chain
    assert [f.name for f in tmp_path.iterdir()] == ["current.json"]  # no temp file left behind


def verdict(rounds, now, switched_at=0.0):
    """Watch from t=0 for 600 s, normal 0.80, stale after 300 s without a complete round."""
    return watch_verdict(rounds, expected=0.80, started=0.0, switched_at=switched_at, now=now, window=600,
                         stale_after=300)


def test_watch_rolls_back_after_two_low_rounds_in_a_row():
    v, why = verdict([(60, 0.60), (120, 0.62)], now=130)  # the alert's rule: two rounds > 10 points below
    assert v == "rollback" and "2 rounds in a row" in why


def test_watch_one_low_round_is_not_enough():
    assert verdict([(60, 0.60), (120, 0.80)], now=130)[0] == "wait"
    assert verdict([(60, 0.70), (120, 0.70)], now=130)[0] == "wait"  # exactly 10 points below: not "more than"


def test_watch_rolls_back_when_the_canary_goes_quiet():
    v, why = verdict([(60, 0.80)], now=400)  # 340 s since the last complete round
    assert v == "rollback" and "no complete round" in why
    v, why = verdict([], now=400, switched_at=None)  # the canary never switched to it
    assert v == "rollback" and "no complete round" in why


def test_watch_passes_after_the_window_with_enough_good_rounds():
    assert verdict([(200, 0.80), (400, 0.76), (590, 0.82)], now=600) == ("pass", "3 rounds, none low twice in a row")


def test_watch_without_enough_rounds_rolls_back_to_be_safe():
    v, why = verdict([(500, 0.80)], now=600)
    assert v == "rollback" and "only 1 complete round" in why


def test_watch_waits_inside_the_window():
    assert verdict([(60, 0.80)], now=120)[0] == "wait"


@pytest.fixture
def prometheus():
    """A fake Prometheus query API: prometheus.series = {query: [(labels, value as text)]}."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)["query"][0]
            result = [{"metric": labels, "value": [1000.0, value]} for labels, value in prom.series.get(query, [])]
            out = json.dumps({"status": "success", "data": {"resultType": "vector", "result": result}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    prom = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    prom.series, prom.url = {}, f"http://127.0.0.1:{prom.server_address[1]}"
    threading.Thread(target=prom.serve_forever, daemon=True).start()
    yield prom
    prom.shutdown()


def test_canary_reading_from_prometheus(prometheus):
    prometheus.series = {'canary_model_info{model="run1"}': [({"model": "run1"}, "900")],
                         "canary_accuracy": [({}, "0.84")], "canary_expected_accuracy": [({}, "0.8")],
                         "canary_last_run_timestamp_seconds": [({}, "960")]}
    assert canary_reading(prometheus.url, "run1") == {"switched_at": 900.0, "accuracy": 0.84, "expected": 0.8,
                                                      "last_run": 960.0}


def test_canary_reading_missing_or_unscored_is_none(prometheus):
    prometheus.series = {"canary_accuracy": [({}, "NaN")]}  # not switched yet; a round without a score
    assert canary_reading(prometheus.url, "run1") == {"switched_at": None, "accuracy": None, "expected": None,
                                                      "last_run": None}


def test_add_round_counts_each_of_the_new_models_rounds_once():
    reading = {"switched_at": 900.0, "accuracy": 0.84, "expected": 0.8, "last_run": 960.0}
    rounds = add_round([], reading)
    assert rounds == [(960.0, 0.84)]
    assert add_round(rounds, reading) == rounds  # polled again before the next round: same round
    assert add_round([], {**reading, "last_run": 880.0}) == []  # finished before the switch: the old model's
    assert add_round([], {**reading, "switched_at": None}) == []  # the canary hasn't switched yet
    assert add_round([], {**reading, "accuracy": None}) == []


def test_champion_info_training_pool_falls_back_to_its_data(tmp_path):
    """A retrained champion's own data is a small, skewed fine-tune set: drift reference and the pool
    to pick from stay the full data it was drawn from (pool_data_dir), else its data_dir (v3)."""
    from training.qlora_financial.registry import register
    adapter, full, small = tmp_path / "adapter", tmp_path / "full", tmp_path / "small"
    for d in (adapter, full, small):
        d.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "b", "r": 16, "lora_alpha": 32}))
    for d in (full, small):
        for split in ("train", "val", "test"):
            (d / f"{split}.jsonl").write_text("{}\n")
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps(r) + "\n" for r in answers(range(4))))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # a throwaway database, never the real one
    import mlflow
    mlflow.set_tracking_uri(uri)
    common = dict(adapter_dir=adapter, samples_path=samples, git_commit="a", artifact_root=tmp_path / "art", champion=True)
    register("v3", data_dir=full, **common)
    info = champion_info(uri)
    assert (info["name"], info["data_dir"], info["pool_data_dir"]) == ("v3", str(full), str(full))
    register("run1", data_dir=small, tags={"pool_data_dir": str(full)}, **common)
    info = champion_info(uri)
    assert (info["name"], info["data_dir"], info["pool_data_dir"]) == ("run1", str(small), str(full))
    assert info["adapter"] == str(adapter) and info["samples"] == str(samples)
