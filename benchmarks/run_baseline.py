"""Concurrent runner for benchmarks/baseline_prompts.jsonl against a vLLM
OpenAI-compatible /v1/completions endpoint. Stdlib-only (urllib + threads),
no extra pip installs needed.

Usage:
    python3 benchmarks/run_baseline.py --host 89.169.102.146 \
        --model meta-llama/Meta-Llama-3-8B-Instruct --concurrency 50

    # Quick smoke test against a handful of prompts first:
    python3 benchmarks/run_baseline.py --host <ip> --limit 10 --concurrency 5
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "baseline_prompts.jsonl"


def load_prompts(path: Path, limit: int | None):
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_S = 0.3


def send_request(endpoint: str, model: str, record: dict, timeout: float) -> dict:
    payload = {
        "model": model,
        "prompt": record["prompt"],
        "max_tokens": record["max_tokens"],
    }
    body = json.dumps(payload).encode("utf-8")
    start = time.monotonic()
    last_error = None

    for attempt in range(TRANSIENT_RETRIES + 1):
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latency = time.monotonic() - start
            usage = data.get("usage", {})
            return {
                "id": record["id"],
                "category": record["category"],
                "status": "ok",
                "latency_s": round(latency, 4),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except urllib.error.HTTPError as e:
            # Server responded with a real error (4xx/5xx) — not a transient
            # connection issue, so don't retry, surface it as-is.
            latency = time.monotonic() - start
            return {
                "id": record["id"],
                "category": record["category"],
                "status": "error",
                "latency_s": round(latency, 4),
                "error": f"HTTP {e.code}: {e.reason}",
            }
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_error = str(e)
            if attempt < TRANSIENT_RETRIES:
                time.sleep(TRANSIENT_BACKOFF_S * (attempt + 1))
                continue

    latency = time.monotonic() - start
    return {
        "id": record["id"],
        "category": record["category"],
        "status": "error",
        "latency_s": round(latency, 4),
        "error": f"{last_error} (after {TRANSIENT_RETRIES} retries)",
    }


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="vLLM server host/IP (no scheme)")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE))
    ap.add_argument("--concurrency", default=50, type=int)
    ap.add_argument("--limit", default=None, type=int, help="Only run the first N prompts")
    ap.add_argument("--timeout", default=120.0, type=float, help="Per-request timeout (s)")
    ap.add_argument("--output", default=None, help="Output JSONL path (default: benchmarks/results_<ts>.jsonl)")
    args = ap.parse_args()

    endpoint = f"http://{args.host}:{args.port}/v1/completions"
    prompts_path = Path(args.prompts_file)
    records = load_prompts(prompts_path, args.limit)
    if not records:
        print(f"No prompts loaded from {prompts_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path(__file__).parent / f"results_{int(time.time())}.jsonl"

    print(f"Endpoint:    {endpoint}")
    print(f"Prompts:     {len(records)} (from {prompts_path.name})")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output:      {out_path}")
    print()

    results = []
    wall_start = time.monotonic()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(send_request, endpoint, args.model, record, args.timeout): record
            for record in records
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == len(records):
                print(f"  {completed}/{len(records)} done...")

    wall_time = time.monotonic() - wall_start

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = [r["latency_s"] for r in ok]
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    total_prompt_tokens = sum(r["prompt_tokens"] or 0 for r in ok)

    print()
    print("==== Summary ====")
    print(f"Wall time:          {wall_time:.1f}s")
    print(f"Requests:            {len(results)} ({len(ok)} ok, {len(errors)} error)")
    if latencies:
        print(f"Latency p50/p95/p99: {percentile(latencies,50):.2f}s / {percentile(latencies,95):.2f}s / {percentile(latencies,99):.2f}s")
        print(f"Latency mean:        {statistics.mean(latencies):.2f}s")
    print(f"Prompt tokens:       {total_prompt_tokens}")
    print(f"Completion tokens:   {total_completion_tokens}")
    if wall_time > 0:
        print(f"Throughput:          {total_completion_tokens / wall_time:.1f} completion tok/s (aggregate)")
        print(f"Request rate:        {len(ok) / wall_time:.2f} req/s")
    if errors:
        print()
        print(f"First few errors:")
        for r in errors[:5]:
            print(f"  {r['id']}: {r['error']}")
    print()
    print(f"Per-request results written to {out_path}")


if __name__ == "__main__":
    main()