"""Locust load test for an OpenAI-compatible POST /v1/completions endpoint
-- points at lightserve's own server (server/app.py), a real vLLM server
(see nebius_setup_commands.txt's `docker run vllm/vllm-openai` command), or
a real SGLang server (see benchmarks/README.md's "Running it yourself"),
with the same task shape, so a side-by-side comparison at a given
concurrency level is an apples-to-apples "same request pattern, same
hardware, different engine" comparison. vLLM and SGLang both speak the same
text-in/text-out OpenAI completions shape, so `--target vllm` covers both --
see below.

Usage -- headless, at the concurrency levels this benchmark cares about
(run each target against its own server, one at a time, each on its own
port -- see benchmarks/README.md's "Running it yourself" for all three):

    locust -f benchmarks/locustfile.py --host http://localhost:8001 \\
        --headless --users 10 --spawn-rate 10 --run-time 120s \\
        --target lightserve --csv benchmarks/locust_results/lightserve_10

    locust -f benchmarks/locustfile.py --host http://localhost:8000 \\
        --headless --users 50 --spawn-rate 50 --run-time 120s \\
        --target vllm --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --csv benchmarks/locust_results/vllm_50

Or interactively: drop --headless/--users/--spawn-rate/--run-time and open
http://localhost:8089 to drive it from the web UI instead (--target and
--model still need to be passed on the command line -- Locust exposes
custom CLI args, not custom UI fields).

See benchmarks/README.md for the real run matrix and results (10/30/50
concurrent users, all three engines) and how to read the resulting --csv
files.

Why --target exists at all, and why there are two prompt files: lightserve
and vLLM/SGLang don't accept the same prompt shape. vLLM and SGLang both
have a real tokenizer and take text (`benchmarks/baseline_prompts.jsonl`);
lightserve has none (see server/schemas.py's module docstring) and takes
token ids (`benchmarks/token_prompts.jsonl`, generate_token_prompts.py's
length-matched companion to the same file). --target picks which one to
load and how to shape the request body -- everything else about the task
(closed-loop request pattern, streaming handling, pass/fail rule) is
identical between the two, which is the point.
"""
import json
import random
import time
from pathlib import Path

from locust import HttpUser, between, events, task

_DIR = Path(__file__).parent


def _load_jsonl(path: Path) -> list:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@events.init_command_line_parser.add_listener
def _add_args(parser):
    parser.add_argument(
        "--target", choices=["lightserve", "vllm"], default="lightserve",
        help="Which server's payload shape to send -- see module docstring.",
    )
    parser.add_argument(
        "--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Passed through as the request body's `model` field; neither server currently rejects a mismatched one.",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Send stream:true and read the SSE body to completion. Locust's own latency numbers then stop "
             "at the response headers; use --samples-out for time to first and last token.",
    )
    parser.add_argument(
        "--samples-out",
        help="Append one JSON line per request (t_start, ttft_s, total_s, tokens, ok) to this file, for "
             "benchmarks/ab/: every request's own timings, not just Locust's percentiles. Needs --stream.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Each simulated user picks prompts from its own random generator (seed + its number), so every "
             "run sends the same prompts in the same order -- for benchmarks/ab/'s rounds. Default: unseeded.",
    )


_samples = None  # the --samples-out file, opened at test start


@events.test_start.add_listener
def _open_samples(environment, **_):
    global _samples
    path = environment.parsed_options.samples_out
    if path:
        if not environment.parsed_options.stream:
            raise SystemExit("--samples-out needs --stream (time to first token comes from the stream)")
        _samples = open(path, "a", buffering=1)


class CompletionsUser(HttpUser):
    # Closed-loop, no think time: fire the next request the moment the last
    # one completes. This benchmark measures how each server behaves with N
    # requests continuously in flight (its saturation/queuing behavior under
    # sustained concurrent load), not a simulated real-user arrival rate --
    # Locust's own --users *is* that N here, not a population of users each
    # pacing themselves.
    wait_time = between(0, 0)

    _started = 0  # users started so far: each seeded user's own number

    def on_start(self):
        seed = self.environment.parsed_options.seed
        self.rng = random if seed is None else random.Random(seed + CompletionsUser._started)
        CompletionsUser._started += 1
        target = self.environment.parsed_options.target
        prompts_file = "token_prompts.jsonl" if target == "lightserve" else "baseline_prompts.jsonl"
        self.records = _load_jsonl(_DIR / prompts_file)
        self.model = self.environment.parsed_options.model
        self.stream = self.environment.parsed_options.stream

    @task
    def completion(self):
        record = self.rng.choice(self.records)
        payload = {
            "model": self.model,
            "prompt": record["prompt"],
            "max_tokens": record["max_tokens"],
            "stream": self.stream,
        }
        t_start, t0 = time.time(), time.perf_counter()
        ttft, tokens, finished = None, 0, False
        with self.client.post(
            "/v1/completions", json=payload, stream=self.stream, catch_response=True,
            name="/v1/completions [stream]" if self.stream else "/v1/completions",
        ) as resp:
            ok = resp.status_code == 200
            if not ok:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
            else:
                if self.stream:
                    # Drain the SSE body (the request isn't over until it ends).
                    # Locust's own timer still stops at the response headers
                    # with stream=True (measured: ~4 ms vs 2-4 s here), so its
                    # percentiles are time-to-headers; --samples-out has the
                    # full times.
                    # Each non-empty text chunk is one token (lightserve and vLLM
                    # both stream one per chunk); the first one's arrival is TTFT.
                    # Complete only with a finish_reason: a "200 OK" stream can
                    # still end early (empty or cut off) without one.
                    for line in resp.iter_lines():
                        if line.startswith(b"data: {"):
                            choice = json.loads(line[6:])["choices"][0]
                            if choice.get("text"):
                                tokens += 1
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                            finished = finished or bool(choice.get("finish_reason"))
                resp.success()
        if _samples:
            _samples.write(json.dumps({"t_start": t_start, "ttft_s": ttft, "total_s": time.perf_counter() - t0,
                                       "tokens": tokens, "ok": ok and finished and ttft is not None}) + "\n")
