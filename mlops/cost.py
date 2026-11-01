"""What the GPUs cost: every GPU job (a fine-tune, a scoring run, a node session) writes a start
and an end line to a ledger (gpu_jobs.jsonl); this adds them up and serves the numbers to
Prometheus for the Cost dashboard and the budget alert. The hourly rate is one setting: GPU
time x rate, so it should match the cloud bill for the same hours (step 9 checks)."""
import argparse
import contextlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client.core import GaugeMetricFamily


def record(ledger, event: str, job_id: str, kind: str, name: str, gpus: int, at: float = None, **extra) -> None:
    """Append one start/end line. Appends are small single writes, so jobs can share the file."""
    line = {"event": event, "id": job_id, "kind": kind, "name": name, "gpus": gpus,
            "at": time.time() if at is None else at, **extra}
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps(line) + "\n")


@contextlib.contextmanager
def gpu_job(ledger, kind: str, name: str, gpus: int = 1):
    """Record a job's start and end; a failed job is still recorded (its GPU time was spent)."""
    job_id = uuid.uuid4().hex
    record(ledger, "start", job_id, kind, name, gpus)
    failed = True
    try:
        yield
        failed = False
    finally:
        record(ledger, "end", job_id, kind, name, gpus, failed=failed)


def start_session(ledger, name: str, gpus: int = 1, at: float = None) -> None:
    """A node/cluster session starts (its GPUs are paid for from now)."""
    record(ledger, "start", f"session-{name}-{uuid.uuid4().hex[:8]}", "session", name, gpus, at)


def stop_session(ledger, name: str, at: float = None) -> None:
    """Close the open session of that name (a name used again later is a new session)."""
    events = [json.loads(line) for line in Path(ledger).read_text().splitlines()] if Path(ledger).exists() else []
    ended = {e["id"] for e in events if e["event"] == "end"}
    open_ = [e for e in events if e["event"] == "start" and e["kind"] == "session" and e["name"] == name
             and e["id"] not in ended]
    if not open_:
        raise ValueError(f"no open session named {name}")
    s = open_[-1]
    record(ledger, "end", s["id"], "session", name, s["gpus"], at)


def summarize(ledger, rate: float, now: float) -> dict:
    """Per job: GPU-seconds (time x GPUs) and cost; a job not ended yet counts up to `now`.
    month_cost = jobs that ended (or are still running) in now's calendar month (UTC)."""
    starts, ends = {}, {}
    if Path(ledger).exists():
        for e in map(json.loads, Path(ledger).read_text().splitlines()):
            (starts if e["event"] == "start" else ends)[e["id"]] = e
    month = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m")
    jobs, month_cost = [], 0.0
    for job_id, s in starts.items():
        end = ends[job_id]["at"] if job_id in ends else now
        gpu_seconds = (end - s["at"]) * s["gpus"]
        cost = gpu_seconds / 3600 * rate
        jobs.append({"kind": s["kind"], "name": s["name"], "gpu_seconds": gpu_seconds, "cost": cost,
                     "open": job_id not in ends})
        if datetime.fromtimestamp(end, timezone.utc).strftime("%Y-%m") == month:
            month_cost += cost
    return {"jobs": jobs, "month_cost": month_cost}


class LedgerCollector:
    """Reads the ledger on every scrape, so new jobs show up without a restart."""
    def __init__(self, ledger, rate: float, budget: float, now=time.time):
        self.ledger, self.rate, self.budget, self.now = ledger, rate, budget, now

    def collect(self):
        s = summarize(self.ledger, self.rate, self.now())
        secs = GaugeMetricFamily("gpu_job_seconds", "GPU-seconds (time x GPUs) per job; still running = up to now",
                                 labels=["kind", "name"])
        cost = GaugeMetricFamily("gpu_job_cost_dollars", "Cost per job at the hourly rate", labels=["kind", "name"])
        totals = {}  # one series per (kind, name): a retried task's tries add up
        for j in s["jobs"]:
            t = totals.setdefault((j["kind"], j["name"]), [0.0, 0.0])
            t[0], t[1] = t[0] + j["gpu_seconds"], t[1] + j["cost"]
        for labels, (gpu_seconds, dollars) in totals.items():
            secs.add_metric(list(labels), gpu_seconds)
            cost.add_metric(list(labels), dollars)
        yield secs
        yield cost
        yield GaugeMetricFamily("gpu_cost_month_dollars", "GPU cost this calendar month (UTC)", value=s["month_cost"])
        yield GaugeMetricFamily("gpu_budget_month_dollars", "The monthly GPU budget", value=self.budget)
        yield GaugeMetricFamily("gpu_hourly_rate_dollars", "Price of one GPU-hour", value=self.rate)


def main():
    ap = argparse.ArgumentParser(description="GPU cost: record a node session, or serve the ledger's numbers.")
    ap.add_argument("action", choices=["start", "stop", "serve"])
    ap.add_argument("--ledger", default="mlops/runs/gpu_jobs.jsonl")
    ap.add_argument("--name", help="start/stop: the session's name, e.g. VM-1")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--rate", type=float, help="serve: price of one GPU-hour in dollars (from your bill)")
    ap.add_argument("--budget", type=float, default=100.0, help="serve: monthly budget in dollars")
    ap.add_argument("--port", type=int, default=9102)
    args = ap.parse_args()
    if args.action in ("start", "stop"):
        if not args.name:
            ap.error("--name is needed")
        if args.action == "start":
            start_session(args.ledger, args.name, args.gpus)
        else:
            stop_session(args.ledger, args.name)
        print(f"session {args.name}: {args.action} recorded in {args.ledger}")
        return
    if args.rate is None:
        ap.error("--rate is needed (dollars per GPU-hour, from your bill)")
    from prometheus_client import REGISTRY, start_http_server
    REGISTRY.register(LedgerCollector(args.ledger, args.rate, args.budget))
    start_http_server(args.port)
    print(f"cost numbers for {args.ledger} on :{args.port}/metrics (rate ${args.rate}/GPU-hour, budget ${args.budget})",
          flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
