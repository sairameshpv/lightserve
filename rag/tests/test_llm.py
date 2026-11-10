import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rag.llm import chat


@pytest.fixture
def server():
    """A fake vLLM: records each request body, answers with a scripted reply (or an error)."""
    seen, script = [], {"status": 200, "text": "Answer: -2%"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = {"choices": [{"message": {"role": "assistant", "content": script["text"]}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 7}}
            data = json.dumps(body if script["status"] == 200 else {"error": "bad"}).encode()
            self.send_response(script["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", seen, script
    httpd.shutdown()


def test_chat_sends_the_settings_and_returns_text_tokens_time(server):
    url, seen, _ = server
    messages = [{"role": "user", "content": "What was Apple's revenue growth in 2019?"}]
    r = chat(messages, model="v3", base_url=url, temperature=0.7, top_p=0.95, seed=3, stop=["\nResult:"], max_tokens=256)
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body == {"model": "v3", "messages": messages, "temperature": 0.7, "top_p": 0.95, "seed": 3,
                    "stop": ["\nResult:"], "max_tokens": 256}
    assert (r["text"], r["prompt_tokens"], r["completion_tokens"]) == ("Answer: -2%", 120, 7)
    assert 0 <= r["seconds"] < 5


def test_chat_leaves_out_unset_seed_and_stop(server):
    url, seen, _ = server
    chat([{"role": "user", "content": "hi"}], model="m", base_url=url)
    assert "seed" not in seen[0][1] and "stop" not in seen[0][1]
    assert seen[0][1]["temperature"] == 0.0     # greedy unless asked


def test_chat_raises_on_a_server_error(server):
    url, _, script = server
    script["status"] = 500
    with pytest.raises(RuntimeError, match="500"):
        chat([{"role": "user", "content": "hi"}], model="m", base_url=url)


def test_a_dropped_connection_is_retried_once(server, monkeypatch):
    """A burst of connections can be dropped before the server reads them (seen with a small queue);
    one retry. A server that answers with an error is not retried: that is a real problem."""
    import urllib.request
    url, seen, _ = server
    real, drops = urllib.request.urlopen, []

    def flaky(*a, **k):
        if len(drops) < 1:
            drops.append(1)
            raise ConnectionResetError(54, "Connection reset by peer")
        return real(*a, **k)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert chat([{"role": "user", "content": "hi"}], model="m", base_url=url)["text"] == "Answer: -2%"
    assert len(seen) == 1


def test_two_dropped_connections_raise(server, monkeypatch):
    import urllib.request
    url, _, _ = server
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(ConnectionResetError(54, "reset")))
    with pytest.raises(ConnectionResetError):
        chat([{"role": "user", "content": "hi"}], model="m", base_url=url)
