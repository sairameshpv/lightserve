"""cost.py: the GPU-job ledger, what it adds up to, and the numbers it serves to Prometheus.
Made-up jobs only."""
import json
from datetime import datetime, timezone

import pytest
from prometheus_client import CollectorRegistry

from mlops.cost import LedgerCollector, gpu_job, record, summarize


def t(day, hour=0, month=10):
    return datetime(2026, month, day, hour, tzinfo=timezone.utc).timestamp()


def test_summarize_gpu_hours_and_cost_per_job_and_month(tmp_path):
    ledger = tmp_path / "gpu_jobs.jsonl"
    record(ledger, "start", "j1", "finetune", "run1", gpus=1, at=t(30, 10))
    record(ledger, "end", "j1", "finetune", "run1", gpus=1, at=t(30, 11))  # 1 GPU-hour
    record(ledger, "start", "j2", "session", "VM-1", gpus=2, at=t(31, 23))
    record(ledger, "end", "j2", "session", "VM-1", gpus=2, at=t(1, 0, month=11))  # 2 GPU-hours, ends in November
    s = summarize(ledger, rate=2.0, now=t(2, 0, month=11))
    assert s["jobs"] == [{"kind": "finetune", "name": "run1", "gpu_seconds": 3600.0, "cost": 2.0, "open": False},
                         {"kind": "session", "name": "VM-1", "gpu_seconds": 7200.0, "cost": 4.0, "open": False}]
    assert s["month_cost"] == 4.0  # a job counts in the month it ended (November)


def test_an_open_job_counts_up_to_now(tmp_path):
    ledger = tmp_path / "gpu_jobs.jsonl"
    record(ledger, "start", "j1", "session", "VM-2", gpus=1, at=t(5, 10, month=11))
    s = summarize(ledger, rate=1.0, now=t(5, 10, month=11) + 1800)
    assert s["jobs"] == [{"kind": "session", "name": "VM-2", "gpu_seconds": 1800.0, "cost": 0.5, "open": True}]
    assert s["month_cost"] == 0.5


def test_no_ledger_yet_is_zero(tmp_path):
    assert summarize(tmp_path / "none.jsonl", rate=1.0, now=t(1)) == {"jobs": [], "month_cost": 0.0}


def test_gpu_job_records_start_and_end_even_when_the_job_fails(tmp_path):
    ledger = tmp_path / "gpu_jobs.jsonl"
    with gpu_job(ledger, "score", "run1", gpus=1):
        pass
    with pytest.raises(RuntimeError):
        with gpu_job(ledger, "finetune", "run2", gpus=1):
            raise RuntimeError("out of memory")  # the GPU time was still spent
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(e["event"], e["name"]) for e in events] == [("start", "run1"), ("end", "run1"),
                                                          ("start", "run2"), ("end", "run2")]
    assert events[3]["failed"] is True and len({e["id"] for e in events}) == 2


def test_collector_serves_the_numbers(tmp_path):
    ledger = tmp_path / "gpu_jobs.jsonl"
    record(ledger, "start", "j1", "session", "VM-1", gpus=1, at=t(30, 10))
    record(ledger, "end", "j1", "session", "VM-1", gpus=1, at=t(30, 12))
    registry = CollectorRegistry()
    registry.register(LedgerCollector(ledger, rate=1.5, budget=100.0, now=lambda: t(31)))
    value = registry.get_sample_value
    assert value("gpu_job_seconds", {"kind": "session", "name": "VM-1"}) == 7200.0
    assert value("gpu_job_cost_dollars", {"kind": "session", "name": "VM-1"}) == 3.0
    assert value("gpu_cost_month_dollars") == 3.0
    assert value("gpu_budget_month_dollars") == 100.0 and value("gpu_hourly_rate_dollars") == 1.5


def test_a_session_name_used_twice_is_two_sessions(tmp_path):
    from mlops.cost import start_session, stop_session
    ledger = tmp_path / "gpu_jobs.jsonl"
    for hour in (10, 14):  # VM-1 run twice, an hour each
        start_session(ledger, "VM-1", gpus=1, at=t(30, hour))
        stop_session(ledger, "VM-1", at=t(30, hour + 1))
    s = summarize(ledger, rate=1.0, now=t(31))
    assert [j["gpu_seconds"] for j in s["jobs"]] == [3600.0, 3600.0] and s["month_cost"] == 2.0
    with pytest.raises(ValueError, match="no open session"):
        stop_session(ledger, "VM-1")


def test_a_retried_job_is_one_series_with_both_tries(tmp_path):
    ledger = tmp_path / "gpu_jobs.jsonl"
    for job_id, start in (("a", t(30, 10)), ("b", t(30, 12))):  # the task failed once, then ran again
        record(ledger, "start", job_id, "finetune", "run1", gpus=1, at=start)
        record(ledger, "end", job_id, "finetune", "run1", gpus=1, at=start + 1800)
    registry = CollectorRegistry()
    registry.register(LedgerCollector(ledger, rate=1.0, budget=100.0, now=lambda: t(31)))
    samples = [s for m in registry.collect() if m.name == "gpu_job_seconds" for s in m.samples]
    assert [(s.labels, s.value) for s in samples] == [({"kind": "finetune", "name": "run1"}, 3600.0)]


def test_month_total_counts_sessions_only_jobs_inside_them_are_a_breakdown(tmp_path):
    """The bill is for the node's time (the session); a fine-tune on that node is part of it, not extra."""
    ledger = tmp_path / "gpu_jobs.jsonl"
    record(ledger, "start", "s", "session", "VM-1", gpus=1, at=t(30, 10))
    record(ledger, "start", "f", "finetune", "run1", gpus=1, at=t(30, 10))
    record(ledger, "end", "f", "finetune", "run1", gpus=1, at=t(30, 10) + 600)
    record(ledger, "end", "s", "session", "VM-1", gpus=1, at=t(30, 11))
    s = summarize(ledger, rate=1.0, now=t(31))
    assert s["month_cost"] == 1.0  # the session's hour, not 1 h + 10 min
    assert [j["cost"] for j in s["jobs"]] == [1.0, 600 / 3600]
