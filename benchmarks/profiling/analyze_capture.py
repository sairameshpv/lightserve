"""Turn a raw Session A (nsys, CUDA graphs on) capture into the top-5-kernels
tables for prefill vs decode.

Reads traces/session_a_cuda_gpu_trace.csv (produced by `nsys stats --report
cuda_gpu_trace --format csv` — see run_profiling.sh / benchmarks/profiling/
report.md for how it's generated) and writes kernel_summary_prefill.csv /
kernel_summary_decode.csv next to it.

Segmentation approach (same one used to hand-analyze the first capture):
cluster kernel launches into bursts separated by >50ms idle gaps. Given the
probe firing order baked into run_profiling.sh — warmup, then prefill-probe,
then decode-probe, immediately followed by `nsys stop` with nothing else
hitting the server in between — the decode-probe is reliably the *last*
burst in the whole capture (nothing is traced after it), and the
prefill-probe is the burst immediately before it. (Picking the *largest*
burst instead doesn't work: CUDA graph capture during model startup produces
an even bigger, unrelated burst of kernels than 200 decode steps do.)
The prefill/warmup ambiguity is resolved using the flash-attention kernel's
GrdX grid dimension, which scales with prompt length (>1 for the 140-token
prefill-probe, ==1 for the single-token warmup/decode steps) — verified
against the manually-produced capture in this session; see report.md's
"Methodology notes" section.

Usage:
    python3 benchmarks/profiling/analyze_capture.py
    python3 benchmarks/profiling/analyze_capture.py --traces-dir /path/to/traces
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

GAP_NS = 50_000_000  # 50ms


def load_trace(path: Path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["Start (ns)"]))
    return rows


def cluster_bursts(rows):
    """Returns a list of (start_idx, end_idx) exclusive slices into `rows`."""
    bursts = []
    burst_start_idx = 0
    cur_end_ns = int(rows[0]["Start (ns)"]) + int(rows[0]["Duration (ns)"])
    for i in range(1, len(rows)):
        start_ns = int(rows[i]["Start (ns)"])
        if start_ns - cur_end_ns > GAP_NS:
            bursts.append((burst_start_idx, i))
            burst_start_idx = i
        cur_end_ns = max(cur_end_ns, start_ns + int(rows[i]["Duration (ns)"]))
    bursts.append((burst_start_idx, len(rows)))
    return bursts


def flash_attn_max_grdx(rows, start_idx, end_idx):
    best = 0
    for r in rows[start_idx:end_idx]:
        if "flash_fwd_splitkv_kernel" in r["Name"] and "combine" not in r["Name"]:
            try:
                best = max(best, int(r["GrdX"]))
            except (KeyError, ValueError):
                pass
    return best


def find_prefill_decode_windows(rows):
    bursts = cluster_bursts(rows)
    if len(bursts) < 2:
        raise ValueError(f"Expected at least 2 bursts (prefill + decode), found {len(bursts)}")

    # decode-probe = the last burst in the capture (nsys stop is called right
    # after it finishes, so nothing else gets traced afterward)
    decode_burst_i = len(bursts) - 1

    if decode_burst_i == 0:
        raise ValueError("Largest (decode) burst has no preceding burst to use as prefill")

    # prefill-probe = immediately preceding burst with a wider flash-attention
    # GrdX than the decode burst's own (decode's is always 1, single token)
    decode_grdx = flash_attn_max_grdx(rows, *bursts[decode_burst_i])
    prefill_burst_i = decode_burst_i - 1
    prefill_grdx = flash_attn_max_grdx(rows, *bursts[prefill_burst_i])
    if prefill_grdx <= decode_grdx:
        raise ValueError(
            f"Burst immediately before decode doesn't look prefill-shaped "
            f"(GrdX {prefill_grdx} <= decode's {decode_grdx}) — probe firing "
            f"order may not match run_profiling.sh's warmup->prefill->decode "
            f"sequence. Inspect session_a_cuda_gpu_trace.csv manually."
        )

    return bursts[prefill_burst_i], bursts[decode_burst_i]


def top5(rows, start_idx, end_idx):
    agg = defaultdict(lambda: [0, 0])
    for r in rows[start_idx:end_idx]:
        agg[r["Name"]][0] += int(r["Duration (ns)"])
        agg[r["Name"]][1] += 1
    total = sum(v[0] for v in agg.values())
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:5]
    return [
        {
            "kernel_name": name,
            "total_time_ms": round(dur / 1e6, 4),
            "pct_of_window": round(100 * dur / total, 2) if total else 0.0,
            "instances": cnt,
        }
        for name, (dur, cnt) in ranked
    ], total


def write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel_name", "total_time_ms", "pct_of_window", "instances"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", default=str(Path(__file__).parent / "traces"))
    ap.add_argument("--output-dir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    output_dir = Path(args.output_dir)
    trace_csv = traces_dir / "session_a_cuda_gpu_trace.csv"
    if not trace_csv.exists():
        raise SystemExit(f"{trace_csv} not found — run a capture first (terraform apply -var enable_profiling=true)")

    rows = load_trace(trace_csv)
    prefill_window, decode_window = find_prefill_decode_windows(rows)

    prefill_top5, prefill_total = top5(rows, *prefill_window)
    decode_top5, decode_total = top5(rows, *decode_window)

    write_csv(output_dir / "kernel_summary_prefill.csv", prefill_top5)
    write_csv(output_dir / "kernel_summary_decode.csv", decode_top5)

    print(f"Prefill window: {prefill_window[1] - prefill_window[0]} kernels, {prefill_total/1e6:.3f}ms total")
    for row in prefill_top5:
        print(f"  {row['pct_of_window']:5.1f}%  {row['total_time_ms']:8.3f}ms  x{row['instances']:<5} {row['kernel_name'][:70]}")

    print(f"\nDecode window: {decode_window[1] - decode_window[0]} kernels, {decode_total/1e6:.3f}ms total")
    for row in decode_top5:
        print(f"  {row['pct_of_window']:5.1f}%  {row['total_time_ms']:8.3f}ms  x{row['instances']:<5} {row['kernel_name'][:70]}")

    print(f"\nWrote {output_dir / 'kernel_summary_prefill.csv'} and {output_dir / 'kernel_summary_decode.csv'}")


if __name__ == "__main__":
    main()