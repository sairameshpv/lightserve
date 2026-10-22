"""server/app.py's request log: one JSON line per finished request on the
`lightserve.requests` logger, carrying the trace id when the request was traced
(so a log line in Loki links to its trace in Tempo)."""
import json
import logging

from server.tests.test_tracing import split, traced_client


def request_lines(caplog) -> list:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "lightserve.requests"]


def test_one_json_line_per_request_with_its_trace_id(caplog):
    client, exporter = traced_client(trace_sample_rate=1.0)
    with caplog.at_level(logging.INFO, logger="lightserve.requests"), client:
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4, "request_id": "r1"})
    [line] = request_lines(caplog)
    root, _ = split(exporter.get_finished_spans())
    assert line["request_id"] == "r1"
    assert line["status"] == "FINISHED_LENGTH_CAPPED" and line["tokens"] == 4
    assert line["trace_id"] == format(root.context.trace_id, "032x")  # the id Tempo shows
    assert 0 < line["ttft_s"] <= line["latency_s"]


def test_untraced_request_still_logged_without_a_trace_id(caplog):
    client, _ = traced_client(trace_sample_rate=0.0, trace_slow_s=10.0)
    with caplog.at_level(logging.INFO, logger="lightserve.requests"), client:
        client.post("/v1/completions", json={"prompt": [1], "max_tokens": 4})
    [line] = request_lines(caplog)
    assert line["trace_id"] is None and line["tokens"] == 4
