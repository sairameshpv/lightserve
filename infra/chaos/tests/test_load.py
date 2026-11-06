"""Tests for infra/chaos/load.py: every request's fate becomes one record, against a fake server
that plays lightserve's streaming endpoint well and badly."""
import io
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from infra.chaos.load import one_request, run


def frame(text, finish=None):
    return "data: " + json.dumps({"choices": [{"text": text, "finish_reason": finish}]}) + "\n\n"


class Fake(BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.mode == "503":
            self.send_response(503), self.end_headers()
            return
        if self.mode == "silent":
            time.sleep(2)     # longer than the test's client limit
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        body = frame("a") + frame("b")
        body += {"ok": frame("", "length"), "abort": frame("", "abort"), "cut": ""}[self.mode]
        self.wfile.write((body + "data: [DONE]\n\n").encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.mark.parametrize("mode, outcome, status", [
    ("ok", "ok", 200),          # tokens and a proper finish
    ("abort", "aborted", 200),  # the server's own timeout: a clean ending, not a failure of the drill
    ("cut", "error", 200),      # a "200 OK" stream that stopped without a finish
    ("503", "error", 503),
])
def test_each_request_becomes_one_record(server, mode, outcome, status):
    srv, url = server
    Fake.mode = mode
    r = one_request(url, timeout_s=5)
    assert (r["outcome"], r["status"]) == (outcome, status)
    assert r["total_s"] >= 0 and r["t_start"] > 0
    if outcome in ("ok", "aborted"):
        assert r["tokens"] == 2 and r["ttft_s"] is not None


def test_no_answer_within_the_limit_is_hung(server):
    srv, url = server
    Fake.mode = "silent"
    r = one_request(url, timeout_s=0.5)
    assert r["outcome"] == "hung" and r["total_s"] < 1.5


def test_a_refused_connection_is_an_error_not_a_crash():
    with socket.socket() as s:      # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = one_request(f"http://127.0.0.1:{port}", timeout_s=1)
    assert r["outcome"] == "error" and r["status"] is None and "refused" in r["error"].lower()


def test_run_writes_one_json_line_per_request_from_every_user(server):
    srv, url = server
    Fake.mode = "ok"
    out = io.StringIO()
    run(url, users=3, duration_s=0.5, out=out, timeout_s=5)
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(rows) >= 3 and all(r["outcome"] == "ok" for r in rows)
    assert {r["user"] for r in rows} == {0, 1, 2}
