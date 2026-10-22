"""server/app.py's per-request trace (OpenTelemetry), captured in memory. One root span
per request, with child spans for its phases: queue (submitted -> first scheduled),
prefill (-> first token), decode (-> last token), respond (-> response sent)."""
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from server.app import create_app
from server.engine_worker import EngineWorker
from server.tests.fake_engine import FakeEngine


def traced_client(**tracing):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    worker = EngineWorker(FakeEngine(), idle_poll_s=0.005, default_timeout_s=5.0)
    return TestClient(create_app(worker, tracer_provider=provider, **tracing)), exporter


def split(spans):
    [root] = [s for s in spans if s.parent is None]
    return root, sorted((s for s in spans if s.parent is not None), key=lambda s: s.start_time)


def test_one_request_one_trace_whose_phases_tile_the_request():
    client, exporter = traced_client(trace_sample_rate=1.0)
    with client:
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4})
    root, phases = split(exporter.get_finished_spans())
    assert [s.name for s in phases] == ["queue", "prefill", "decode", "respond"]
    assert {s.context.trace_id for s in phases} == {root.context.trace_id}
    # Back to back, from the root's start to its end: the phases explain all of its time.
    assert phases[0].start_time == root.start_time and phases[-1].end_time == root.end_time
    assert all(a.end_time == b.start_time for a, b in zip(phases, phases[1:]))
    assert root.attributes["lightserve.status"] == "FINISHED_LENGTH_CAPPED"
    assert root.attributes["lightserve.tokens"] == 4
    assert phases[2].attributes["lightserve.tokens"] == 3  # decode = every token after the first


def test_sampling_keeps_slow_and_failed_requests_and_drops_the_rest():
    client, exporter = traced_client(trace_sample_rate=0.0, trace_slow_s=10.0)
    with client:
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4})  # fast, fine: dropped
        assert exporter.get_finished_spans() == ()
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4, "timeout_s": 0.0})  # aborted: kept
    root, _ = split(exporter.get_finished_spans())
    assert root.attributes["lightserve.status"] == "FINISHED_ABORTED"
    assert root.status.status_code == StatusCode.ERROR

    client, exporter = traced_client(trace_sample_rate=0.0, trace_slow_s=0.0)  # everything counts as slow
    with client:
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4})
    assert split(exporter.get_finished_spans())[0].attributes["lightserve.tokens"] == 4
