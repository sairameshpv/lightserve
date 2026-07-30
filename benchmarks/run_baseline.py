"""Concurrent runner for benchmarks/baseline_prompts.jsonl against a vLLM
OpenAI-compatible /v1/completions endpoint. Stdlib-only (urllib + threads),
no extra pip installs needed.

Usage:
    python3 benchmarks/run_baseline.py --host 89.169.102.146 \
        --model meta-llama/Meta-Llama-3-8B-Instruct --concurrency 50

    # Quick smoke test against a handful of prompts first:
    python3 benchmarks/run_baseline.py --host <ip> --limit 10 --concurrency 5

    # Full baseline, run 3 times sequentially (e.g. to see whether prefix
    # caching speeds up later passes over the same prompt set):
    python3 benchmarks/run_baseline.py --host <ip> --repeats 3
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "baseline_prompts.jsonl"
DEFAULT_README_FILE = Path(__file__).parent / "README.md"


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


def percentile(values: list[float], pct: float):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_pass(records, run_idx, endpoint, model, concurrency, timeout):
    """Runs one full concurrent pass over `records`. Returns (results, wall_time)."""
    print(f"--- Run {run_idx}: {len(records)} prompts, concurrency {concurrency} ---")
    results = []
    wall_start = time.monotonic()
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(send_request, endpoint, model, record, timeout): record
            for record in records
        }
        for future in as_completed(futures):
            result = future.result()
            result["run"] = run_idx
            results.append(result)
            completed += 1
            if completed % 100 == 0 or completed == len(records):
                print(f"  {completed}/{len(records)} done...")

    wall_time = time.monotonic() - wall_start
    return results, wall_time


def summarize(label, results, wall_time):
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = [r["latency_s"] for r in ok]
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    total_prompt_tokens = sum(r["prompt_tokens"] or 0 for r in ok)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    throughput = total_completion_tokens / wall_time if wall_time > 0 else 0.0

    print()
    print(f"==== {label} ====")
    print(f"Wall time:           {wall_time:.1f}s")
    print(f"Requests:            {len(results)} ({len(ok)} ok, {len(errors)} error)")
    if latencies:
        print(f"Latency p50/p95/p99: {p50:.2f}s / {p95:.2f}s / {p99:.2f}s")
        print(f"Latency mean:        {statistics.mean(latencies):.2f}s")
    print(f"Prompt tokens:       {total_prompt_tokens}")
    print(f"Completion tokens:   {total_completion_tokens}")
    if wall_time > 0:
        print(f"Throughput:          {throughput:.1f} completion tok/s (aggregate)")
        print(f"Request rate:        {len(ok) / wall_time:.2f} req/s")
    if errors:
        print()
        print("First few errors:")
        for r in errors[:5]:
            print(f"  {r['id']}: {r['error']}")

    return {
        "wall_time": wall_time,
        "ok": len(ok),
        "errors": len(errors),
        "throughput_tok_s": throughput,
        "p50": p50,
        "p95": p95,
        "p99": p99,
    }


README_TABLE_HEADER = (
    "| Timestamp (UTC) | Host | Model | Prompts | Concurrency | Repeats "
    "| Wall (s) | OK | Err | Throughput (tok/s) | p50 (s) | p95 (s) | p99 (s) | Results file |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def append_readme_row(readme_path: Path, meta: dict, stats: dict):
    is_new = not readme_path.exists()
    p50 = f"{stats['p50']:.2f}" if stats["p50"] is not None else "-"
    p95 = f"{stats['p95']:.2f}" if stats["p95"] is not None else "-"
    p99 = f"{stats['p99']:.2f}" if stats["p99"] is not None else "-"
    row = (
        f"| {meta['timestamp']} | {meta['host']} | {meta['model']} | {meta['prompts']} "
        f"| {meta['concurrency']} | {meta['repeats']} | {stats['wall_time']:.1f} "
        f"| {stats['ok']} | {stats['errors']} | {stats['throughput_tok_s']:.1f} "
        f"| {p50} | {p95} | {p99} | {meta['output']} |\n"
    )
    with readme_path.open("a") as f:
        if is_new:
            f.write("# Baseline benchmark results\n\n")
            f.write("Auto-appended by `run_baseline.py` after each run.\n\n")
            f.write(README_TABLE_HEADER)
        f.write(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="vLLM server host/IP (no scheme)")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS_FILE))
    ap.add_argument("--concurrency", default=50, type=int)
    ap.add_argument("--limit", default=None, type=int, help="Only run the first N prompts")
    ap.add_argument("--timeout", default=120.0, type=float, help="Per-request timeout (s)")
    ap.add_argument("--repeats", default=1, type=int, help="Run the full prompt set this many times sequentially")
    ap.add_argument("--output", default=None, help="Output JSONL path (default: benchmarks/results_<ts>.jsonl)")
    ap.add_argument("--readme-file", default=str(DEFAULT_README_FILE), help="Markdown file to append a results row to")
    ap.add_argument("--skip-readme", action="store_true", help="Don't append a row to the README summary table")
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
    print(f"Repeats:     {args.repeats}")
    print(f"Output:      {out_path}")
    print()

    all_results = []
    pass_summaries = []  # (run_idx, results, wall_time) for the per-run comparison table

    final_stats = None
    for run_idx in range(1, args.repeats + 1):
        results, wall_time = run_pass(records, run_idx, endpoint, args.model, args.concurrency, args.timeout)
        final_stats = summarize(f"Run {run_idx}/{args.repeats}", results, wall_time)
        all_results.extend(results)
        pass_summaries.append((run_idx, results, wall_time))

    with out_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    if args.repeats > 1:
        print()
        print("==== Per-run comparison ====")
        print(f"{'run':>4} {'wall_s':>8} {'ok':>6} {'err':>5} {'tok/s':>10} {'p50_s':>8} {'p95_s':>8}")
        for run_idx, results, wall_time in pass_summaries:
            ok = [r for r in results if r["status"] == "ok"]
            latencies = [r["latency_s"] for r in ok]
            tok = sum(r["completion_tokens"] or 0 for r in ok)
            tput = tok / wall_time if wall_time > 0 else 0
            p50 = percentile(latencies, 50) or 0
            p95 = percentile(latencies, 95) or 0
            print(f"{run_idx:>4} {wall_time:>8.1f} {len(ok):>6} {len(results)-len(ok):>5} {tput:>10.1f} {p50:>8.2f} {p95:>8.2f}")

        combined_wall_time = sum(w for _, _, w in pass_summaries)
        final_stats = summarize(f"Combined (all {args.repeats} runs)", all_results, combined_wall_time)

    print()
    print(f"All per-request results ({len(all_results)} rows, tagged with 'run') written to {out_path}")

    if not args.skip_readme and final_stats is not None:
        meta = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "host": args.host,
            "model": args.model,
            "prompts": len(records),
            "concurrency": args.concurrency,
            "repeats": args.repeats,
            "output": out_path.name,
        }
        readme_path = Path(args.readme_file)
        append_readme_row(readme_path, meta, final_stats)
        print(f"Appended results row to {readme_path}")


if __name__ == "__main__":
    main()