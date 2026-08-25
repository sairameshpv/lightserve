"""Side-by-side comparison of two Locust runs (e.g. lightserve vs vLLM at
the same concurrency) from their `--csv` output files.

Usage, after running locustfile.py twice (see benchmarks/README.md):
    python3 benchmarks/compare_locust_runs.py \\
        --a benchmarks/locust_lightserve_10 --a-label lightserve \\
        --b benchmarks/locust_vllm_10       --b-label vllm

Reads two files per run, both written by Locust itself under the given
`--csv` prefix -- no re-derivation of numbers Locust already computed:
  `<prefix>_stats.csv`         -- final totals and response-time percentiles
                                   (the "Aggregated" row, across all named
                                   requests/tasks).
  `<prefix>_stats_history.csv` -- a Requests/s time series; this script uses
                                   its *last* row as the run's steady-state
                                   throughput snapshot, since `_stats.csv`'s
                                   own Requests/s field reflects the whole
                                   run including ramp-up, not just steady
                                   state.
"""
import argparse
import csv
from pathlib import Path


def _read_aggregated_row(stats_csv: Path) -> dict:
    with stats_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["Name"] == "Aggregated":
                return row
    raise ValueError(f"no 'Aggregated' row found in {stats_csv}")


def _read_last_history_row(history_csv: Path) -> dict:
    with history_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {history_csv}")
    return rows[-1]


def load_run(prefix: str) -> dict:
    stats = _read_aggregated_row(Path(f"{prefix}_stats.csv"))
    history = _read_last_history_row(Path(f"{prefix}_stats_history.csv"))
    return {
        "requests": int(stats["Request Count"]),
        "failures": int(stats["Failure Count"]),
        "p50_ms": float(stats["50%"]),
        "p95_ms": float(stats["95%"]),
        "p99_ms": float(stats["99%"]),
        "max_ms": float(stats["Max Response Time"]),
        "req_s": float(history["Requests/s"]),
    }


ROW_LABELS = [
    ("requests", "Requests", "{:.0f}"),
    ("failures", "Failures", "{:.0f}"),
    ("req_s", "Throughput (req/s)", "{:.2f}"),
    ("p50_ms", "Latency p50 (ms)", "{:.1f}"),
    ("p95_ms", "Latency p95 (ms)", "{:.1f}"),
    ("p99_ms", "Latency p99 (ms)", "{:.1f}"),
    ("max_ms", "Latency max (ms)", "{:.1f}"),
]


def print_comparison(a: dict, a_label: str, b: dict, b_label: str) -> None:
    name_w = max(len(label) for _, label, _ in ROW_LABELS)
    val_w = max(len(a_label), len(b_label), 12)
    print(f"{'metric':<{name_w}}  {a_label:>{val_w}}  {b_label:>{val_w}}")
    print(f"{'-' * name_w}  {'-' * val_w}  {'-' * val_w}")
    for key, label, fmt in ROW_LABELS:
        print(f"{label:<{name_w}}  {fmt.format(a[key]):>{val_w}}  {fmt.format(b[key]):>{val_w}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="--csv prefix of the first run")
    ap.add_argument("--b", required=True, help="--csv prefix of the second run")
    ap.add_argument("--a-label", default="a")
    ap.add_argument("--b-label", default="b")
    args = ap.parse_args()

    a = load_run(args.a)
    b = load_run(args.b)
    print_comparison(a, args.a_label, b, args.b_label)


if __name__ == "__main__":
    main()
