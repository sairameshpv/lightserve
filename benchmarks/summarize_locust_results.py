"""Regenerates the lightserve-vs-vLLM-vs-SGLang results table in
benchmarks/README.md from the committed --csv files in
benchmarks/locust_results/ -- the exact numbers there were produced by this
script against a real run (see that README section for how/when).

Usage:
    python3 benchmarks/summarize_locust_results.py
    python3 benchmarks/summarize_locust_results.py --run-time-s 120 \\
        --dir benchmarks/locust_results \\
        --engines lightserve,vllm,sglang --users 10,30,50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_locust_runs import load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path(__file__).parent / "locust_results"))
    ap.add_argument("--engines", default="lightserve,vllm,sglang")
    ap.add_argument("--users", default="10,30,50")
    ap.add_argument("--run-time-s", type=float, default=120.0,
                     help="Wall-clock window each run used (--run-time passed to locust) -- "
                          "avg req/s is Request Count / this, not the noisy last-row snapshot "
                          "in _stats_history.csv (see compare_locust_runs.py's module docstring).")
    args = ap.parse_args()

    engines = args.engines.split(",")
    users = [int(u) for u in args.users.split(",")]
    base = Path(args.dir)

    print("| engine | users | requests | failures | avg req/s | p50 (s) | p95 (s) | p99 (s) | max (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for engine in engines:
        for u in users:
            r = load_run(str(base / f"{engine}_{u}"))
            avg_req_s = r["requests"] / args.run_time_s
            print(
                f"| {engine} | {u} | {r['requests']} | {r['failures']} | {avg_req_s:.2f} "
                f"| {r['p50_ms'] / 1000:.2f} | {r['p95_ms'] / 1000:.2f} "
                f"| {r['p99_ms'] / 1000:.2f} | {r['max_ms'] / 1000:.2f} |"
            )


if __name__ == "__main__":
    main()
