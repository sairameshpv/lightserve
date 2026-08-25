"""Locust load test for an OpenAI-compatible POST /v1/completions endpoint
-- points at either lightserve's own server (server/app.py) or a real vLLM
server (see nebius_setup_commands.txt's `docker run vllm/vllm-openai`
command) with the same task shape, so a side-by-side comparison at a given
concurrency level is an apples-to-apples "same request pattern, same
hardware, different engine" comparison.

Usage -- headless, at the two concurrency levels this benchmark cares
about (run each --target against its own server, one at a time; both
listen on :8000 by convention -- see nebius_setup_commands.txt and
server/README.md's "Running it" section):

    locust -f benchmarks/locustfile.py --host http://localhost:8000 \\
        --headless --users 10 --spawn-rate 10 --run-time 2m \\
        --target lightserve --csv benchmarks/locust_lightserve_10

    locust -f benchmarks/locustfile.py --host http://localhost:8000 \\
        --headless --users 50 --spawn-rate 50 --run-time 2m \\
        --target vllm --csv benchmarks/locust_vllm_50

Or interactively: drop --headless/--users/--spawn-rate/--run-time and open
http://localhost:8089 to drive it from the web UI instead (--target and
--model still need to be passed on the command line -- Locust exposes
custom CLI args, not custom UI fields).

See benchmarks/README.md for the full run matrix (10 and 50 concurrent
users, both targets) and how to read the resulting --csv files.

Why --target exists at all, and why there are two prompt files: the two
servers don't accept the same prompt shape. vLLM has a real tokenizer and
takes text (`benchmarks/baseline_prompts.jsonl`); lightserve has none (see
server/schemas.py's module docstring) and takes token ids
(`benchmarks/token_prompts.jsonl`, generate_token_prompts.py's
length-matched companion to the same file). --target picks which one to
load and how to shape the request body -- everything else about the task
(closed-loop request pattern, streaming handling, pass/fail rule) is
identical between the two, which is the point.
"""
import json
import random
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
        help="Send stream:true and read the SSE body to completion, so a request's measured latency is "
             "time-to-last-token rather than time-to-first-byte.",
    )


class CompletionsUser(HttpUser):
    # Closed-loop, no think time: fire the next request the moment the last
    # one completes. This benchmark measures how each server behaves with N
    # requests continuously in flight (its saturation/queuing behavior under
    # sustained concurrent load), not a simulated real-user arrival rate --
    # Locust's own --users *is* that N here, not a population of users each
    # pacing themselves.
    wait_time = between(0, 0)

    def on_start(self):
        target = self.environment.parsed_options.target
        prompts_file = "token_prompts.jsonl" if target == "lightserve" else "baseline_prompts.jsonl"
        self.records = _load_jsonl(_DIR / prompts_file)
        self.model = self.environment.parsed_options.model
        self.stream = self.environment.parsed_options.stream

    @task
    def completion(self):
        record = random.choice(self.records)
        payload = {
            "model": self.model,
            "prompt": record["prompt"],
            "max_tokens": record["max_tokens"],
            "stream": self.stream,
        }
        with self.client.post(
            "/v1/completions", json=payload, stream=self.stream, catch_response=True,
            name="/v1/completions [stream]" if self.stream else "/v1/completions",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            if self.stream:
                # Drain the SSE body so Locust's own per-request timer (and
                # therefore the reported latency) covers the whole
                # generation, not just the time to the response headers.
                for _ in resp.iter_lines():
                    pass
            resp.success()
