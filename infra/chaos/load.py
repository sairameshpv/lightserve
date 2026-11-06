"""Steady traffic for the chaos experiments (8j): N users each send one streaming completion after
another, and every request's fate is one JSON line on stdout. Standard library only: it runs in a
pod from the lightserve image, mounted from a ConfigMap.

  python load.py --url http://lightserve.lightserve:8000 --users 4 --duration-s 180

outcome: ok (tokens and a proper finish), aborted (the server's own clean timeout: "abort"),
error (a bad status, a refused/reset connection, or a stream that stopped without a finish),
hung (no answer within --timeout-s, set above the server's 60 s so only a real hang gets here).
"""
import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

# The cluster runs lightserve with --max-model-len 128: a short fixed prompt fits with room to spare.
PROMPT, MAX_TOKENS = list(range(1, 17)), 32


def one_request(url, timeout_s, max_tokens=MAX_TOKENS):
    t_start, t0 = time.time(), time.perf_counter()
    rec = {"t_start": t_start, "ttft_s": None, "total_s": None, "tokens": 0, "outcome": "error",
           "status": None, "error": None}
    body = json.dumps({"prompt": PROMPT, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:   # the limit applies to every wait
            rec["status"] = resp.status
            for line in resp:
                if not line.startswith(b"data: {"):
                    continue
                choice = json.loads(line[6:])["choices"][0]
                if choice.get("text"):
                    rec["tokens"] += 1
                    if rec["ttft_s"] is None:
                        rec["ttft_s"] = time.perf_counter() - t0
                finish = choice.get("finish_reason") or finish
        if finish == "abort":
            rec["outcome"] = "aborted"
        elif finish and rec["tokens"]:
            rec["outcome"] = "ok"
        else:
            rec["error"] = "stream ended without a finish"
    except urllib.error.HTTPError as e:
        rec["status"], rec["error"] = e.code, f"HTTP {e.code}"
    except (socket.timeout, TimeoutError):
        rec["outcome"], rec["error"] = "hung", f"no answer within {timeout_s} s"
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            rec["outcome"] = "hung"
        rec["error"] = str(reason)
    rec["total_s"] = time.perf_counter() - t0
    return rec


def run(url, users, duration_s, out, timeout_s, max_tokens=MAX_TOKENS):
    """Each user loops until the time is up; the last requests finish (or hang out) after it."""
    lock, stop = threading.Lock(), time.time() + duration_s

    def user(i):
        while time.time() < stop:
            rec = one_request(url, timeout_s, max_tokens)
            rec["user"] = i
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()

    threads = [threading.Thread(target=user, args=(i,)) for i in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--timeout-s", type=float, default=90.0, help="above the server's own 60 s")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    run(a.url, a.users, a.duration_s, sys.stdout, a.timeout_s, a.max_tokens)


if __name__ == "__main__":
    main()
