"""canary.py: the fixed canary questions, how their answers are graded, and one round against
a fake OpenAI-style chat server (a tiny web server in a thread). No GPU."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from prometheus_client import CollectorRegistry

from training.qlora_financial.canary import expected_accuracy, grade, make_metrics, pick, record, run_once


def questions(n):
    return [{"id": i, "messages": [{}, {"content": f"q{i}"}], "gold": f"Answer: {i}"} for i in range(n)]


def test_pick_is_the_same_50_every_time():
    qs = questions(300)
    first = pick(qs, n=50, seed=0)
    assert len(first) == 50 and len({q["id"] for q in first}) == 50
    assert first == pick(qs, n=50, seed=0) and first != pick(qs, n=50, seed=1)


def test_grade_uses_the_calculator_and_the_fair_checker():
    qs = [{"gold": "Step 1: divide 8475 and 22110.\nAnswer: 0.38331"}, {"gold": "Answer: 0.06625"},
          {"gold": "2018"}, {"gold": "Answer: 5"}]
    answers = ["Step 1: divide 8475 and 22110.\nAnswer: 0.4",  # wrong number, right step: the calculator fixes it
               "Answer: 0.0665",  # 0.4% off: right under the old 1% checker, wrong under the fair one
               "2018",
               "I don't know"]
    assert grade(qs, answers) == 0.5


@pytest.fixture
def server():
    """A fake chat server; set server.reply(prompt) -> answer. Records the model names asked for."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            srv.models.add(body["model"])
            out = json.dumps({"choices": [{"message": {"content": srv.reply(body["messages"][-1]["content"])}}]})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *args):  # keep test output quiet
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 64  # the default 5 refuses some of the canary's 8 parallel connections (macOS: Errno 22)

    srv = Server(("127.0.0.1", 0), Handler)
    srv.models, srv.url = set(), f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_run_once_right_and_wrong_answers(server):
    qs = questions(10)
    server.reply = lambda prompt: "Answer: " + prompt[1:]  # q7 -> "Answer: 7": all right
    assert run_once(server.url, "v3", qs) == {"accuracy": 1.0, "answered": 10, "errors": 0}
    server.reply = lambda prompt: "Answer: 999"
    assert run_once(server.url, "v3", qs) == {"accuracy": 0.0, "answered": 10, "errors": 0}
    assert server.models == {"v3"}


def test_run_once_server_down_counts_errors_without_crashing():
    assert run_once("http://127.0.0.1:9", "v3", questions(3)) == {"accuracy": None, "answered": 0, "errors": 3}


def test_expected_accuracy_is_the_champions_saved_score_on_the_same_questions():
    saved = [{"id": i, "greedy": "Answer: 0" if i < 5 else f"Answer: {i}"} for i in range(10)]  # wrong on 1-4
    assert expected_accuracy(saved, questions(10)) == 0.6
    assert expected_accuracy(saved, questions(10)[5:]) == 1.0  # only the canary's questions count


def test_record_sets_the_metrics():
    registry = CollectorRegistry()
    metrics = make_metrics(registry, expected=0.8)
    assert math.isnan(registry.get_sample_value("canary_accuracy"))  # no round yet: not 0, which would look broken
    record(metrics, {"accuracy": 0.7, "answered": 48, "errors": 2}, now=1000.0)
    value = registry.get_sample_value
    assert value("canary_accuracy") == 0.7
    record(metrics, {"accuracy": None, "answered": 0, "errors": 50}, now=1300.0)  # server down
    assert value("canary_expected_accuracy") == 0.8
    # no score, not the last one: a bad round followed by an outage mustn't keep "low" up
    assert math.isnan(value("canary_accuracy"))
    assert value("canary_errors_total") == 52
    assert value("canary_last_run_timestamp_seconds") == 1000.0  # so the canary goes stale
