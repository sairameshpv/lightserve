"""Benchmark: fused_bias_relu (one Triton kernel) vs the naive two-op PyTorch
path (x + bias, then relu -- two separate kernel launches, x+bias written to
HBM and read back for the relu pass).

The fusion doesn't save FLOPs (there are almost none here -- this op is pure
memory traffic), it saves an HBM round-trip: naive does 2 reads + 2 writes of
an [M,N] tensor (read x/write tmp, read tmp/write out) vs fused's 1 read +
1 write. So the win to look for is in achieved GB/s and wall time, not TFLOPS.

    python3 benchmark_fused_bias_relu.py

Meant to run on the same box as the correctness tests (needs a real CUDA GPU
-- Triton has no CPU backend).
"""
import json
import statistics
import sys

import torch

from fused_bias_relu import fused_bias_relu

DEVICE = "cuda"
WARMUP = 15
ITERS = 100
REPEATS = 5  # take the median of this many timed blocks

SHAPES = [(128, 128), (512, 512), (1024, 1024), (4096, 4096), (4096, 14336), (8192, 8192)]
DTYPES = [torch.float32, torch.bfloat16]


def time_block(fn, iters=ITERS, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    per_call_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_call_ms.append(start.elapsed_time(end) / iters)
    return statistics.median(per_call_ms)


def naive_bias_relu(x, bias):
    return torch.relu(x + bias)


def bytes_moved(M, N, dtype_bytes):
    # fused: read x [M,N] + read bias [N] + write out [M,N]
    fused = dtype_bytes * (M * N + N + M * N)
    # naive: read x + read bias + write tmp, then read tmp + write out
    naive = dtype_bytes * (M * N + N + M * N) + dtype_bytes * (M * N + M * N)
    return fused, naive


def main():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available"}))
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    results = {"gpu": gpu_name, "torch_version": torch.__version__, "rows": []}

    print(f"{'shape':<16}{'dtype':<12}{'fused ms':>10}{'naive ms':>10}"
          f"{'speedup':>9}{'fused GB/s':>12}{'naive GB/s':>12}")
    print("-" * 81)

    for M, N in SHAPES:
        for dtype in DTYPES:
            torch.manual_seed(0)
            x = torch.randn(M, N, device=DEVICE, dtype=dtype)
            bias = torch.randn(N, device=DEVICE, dtype=dtype)

            # Correctness gate: refuse to report a benchmark for a kernel
            # that doesn't match the reference it's being compared against.
            actual = fused_bias_relu(x, bias)
            expected = naive_bias_relu(x, bias)
            atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

            fused_ms = time_block(lambda: fused_bias_relu(x, bias))
            naive_ms = time_block(lambda: naive_bias_relu(x, bias))

            dtype_bytes = torch.tensor([], dtype=dtype).element_size()
            fused_bytes, naive_bytes = bytes_moved(M, N, dtype_bytes)
            fused_gbs = fused_bytes / (fused_ms / 1000) / 1e9
            naive_gbs = naive_bytes / (naive_ms / 1000) / 1e9

            row = {
                "shape": f"{M}x{N}", "dtype": str(dtype).replace("torch.", ""),
                "fused_ms": round(fused_ms, 5), "naive_ms": round(naive_ms, 5),
                "speedup": round(naive_ms / fused_ms, 3),
                "fused_gbs": round(fused_gbs, 1), "naive_gbs": round(naive_gbs, 1),
            }
            results["rows"].append(row)
            print(f"{row['shape']:<16}{row['dtype']:<12}{row['fused_ms']:>10.4f}"
                  f"{row['naive_ms']:>10.4f}{row['speedup']:>8.2f}x"
                  f"{row['fused_gbs']:>12.1f}{row['naive_gbs']:>12.1f}")

    print(f"\nAll {len(results['rows'])} shape/dtype combos verified allclose "
          f"against torch.relu(x + bias) before timing.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
