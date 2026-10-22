"""server/app.py's GET /metrics (Prometheus text format) against FakeEngine,
which emits tokens 0, 1, 2, ... up to max_tokens -- so every count is exact."""
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from engine.config import CacheConfig
from engine.request import SamplingParams
from server.engine_worker import EngineWorker, StreamEnd
from server.tests.fake_engine import FakeEngine
from server.tests.test_app import make_client


def parse(text: str) -> dict:
    """{(sample name, frozen labels): value} from Prometheus text."""
    return {(s.name, frozenset(s.labels.items())): s.value
            for fam in text_string_to_metric_families(text) for s in fam.samples}


def scrape(client) -> dict:
    return parse(client.get("/metrics").text)


def test_metrics_count_requests_tokens_and_latencies():
    client, _ = make_client()
    with client:
        for _ in range(2):
            client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4})
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 3, "stream": True})
        m = scrape(client)
    assert m[("lightserve_requests_total", frozenset({("status", "FINISHED_LENGTH_CAPPED")}))] == 3
    assert m[("lightserve_generated_tokens_total", frozenset())] == 4 + 4 + 3
    assert m[("lightserve_time_to_first_token_seconds_count", frozenset())] == 3
    assert m[("lightserve_inter_token_seconds_count", frozenset())] == 3 + 3 + 2  # gaps between tokens
    assert m[("lightserve_request_seconds_count", frozenset())] == 3
    assert m[("lightserve_requests_in_flight", frozenset())] == 0


def test_engine_metrics_with_a_tiny_kv_cache_that_forces_preemption():
    """4 blocks of 4 tokens, 3 requests of 4 prompt + 8 output tokens: they can't all fit,
    so the scheduler preempts. Submitted before start(), so the first step sees all 3 and
    the run is deterministic: 18 steps, 3 of them prefill (1 first admission of all 3, then
    2 re-admissions after preemption), 2 preemptions."""
    worker = EngineWorker(FakeEngine(CacheConfig(block_size=4, num_gpu_blocks=4)), idle_poll_s=0.001)
    handles = [worker.submit([1, 2, 3, 4], SamplingParams(max_tokens=8)) for _ in range(3)]
    worker.start()
    for h in handles:
        while not isinstance(h.queue.get(timeout=5), StreamEnd):
            pass
    worker.stop()
    m = parse(generate_latest(worker.registry).decode())
    assert m[("lightserve_step_seconds_count", frozenset({("kind", "prefill")}))] == 3
    assert m[("lightserve_step_seconds_count", frozenset({("kind", "decode")}))] == 15
    assert m[("lightserve_batch_size_sum", frozenset())] == 24  # requests per step, summed over the 18 steps
    assert m[("lightserve_queue_wait_seconds_count", frozenset())] == 3  # first admission only, not re-admissions
    assert m[("lightserve_preemptions_total", frozenset())] == 2
    assert m[("lightserve_requests_total", frozenset({("status", "FINISHED_LENGTH_CAPPED")}))] == 3
    assert m[("lightserve_kv_cache_usage", frozenset())] == 0  # everything freed at the end
    assert m[("lightserve_requests_waiting", frozenset())] == m[("lightserve_requests_running", frozenset())] == 0
