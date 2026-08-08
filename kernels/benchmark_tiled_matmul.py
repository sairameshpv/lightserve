"""Benchmark: tiled_matmul (Triton, kernel 3) vs. torch.matmul (cuBLAS) at 3
square GEMM sizes, in bf16 -- the dtype every peak/roofline number elsewhere
in this repo is quoted in (L40S: 362 TFLOPS dense bf16, see
../benchmarks/profiling/roofline/).

Unlike kernels 1 and 2 (memory-bound, GB/s is the interesting number), GEMM
at these sizes is compute-bound, so the interesting number is achieved
TFLOPS and efficiency relative to both cuBLAS and the hardware's peak.

    python3 benchmark_tiled_matmul.py

First call per shape pays Triton's autotune search cost (sweeping the
configs in tiled_matmul.py) -- warmup iterations happen before any timing.
"""
import json
import statistics
import sys

import torch

from tiled_matmul import matmul

DEVICE = "cuda"
DTYPE = torch.bfloat16
PEAK_BF16_TFLOPS = 362.0  # L40S dense bf16 spec, same constant as the roofline analysis

WARMUP = 10
ITERS = 30
REPEATS = 5

SIZES = [1024, 4096, 8192]  # square M=K=N


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


def main():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available"}))
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    results = {"gpu": gpu_name, "torch_version": torch.__version__, "dtype": "bfloat16", "rows": []}

    print(f"{'size (MxKxN)':<16}{'ours ms':>10}{'cuBLAS ms':>11}"
          f"{'ours TFLOPS':>13}{'cuBLAS TFLOPS':>15}{'efficiency %':>14}{'% of peak':>11}")
    print("-" * 90)

    for size in SIZES:
        M = K = N = size
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)

        actual = matmul(a, b)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=5e-2)

        ours_ms = time_block(lambda: matmul(a, b))
        cublas_ms = time_block(lambda: torch.matmul(a, b))

        flops = 2 * M * K * N
        ours_tflops = flops / (ours_ms / 1000) / 1e12
        cublas_tflops = flops / (cublas_ms / 1000) / 1e12
        efficiency = ours_tflops / cublas_tflops * 100
        pct_peak = ours_tflops / PEAK_BF16_TFLOPS * 100

        row = {
            "size": f"{M}x{K}x{N}",
            "ours_ms": round(ours_ms, 5), "cublas_ms": round(cublas_ms, 5),
            "ours_tflops": round(ours_tflops, 1), "cublas_tflops": round(cublas_tflops, 1),
            "efficiency_pct": round(efficiency, 1), "pct_of_l40s_peak": round(pct_peak, 1),
        }
        results["rows"].append(row)
        print(f"{row['size']:<16}{row['ours_ms']:>10.4f}{row['cublas_ms']:>11.4f}"
              f"{row['ours_tflops']:>13.1f}{row['cublas_tflops']:>15.1f}"
              f"{row['efficiency_pct']:>13.1f}%{row['pct_of_l40s_peak']:>10.1f}%")

    print(f"\nAll {len(results['rows'])} sizes verified allclose against torch.matmul before timing.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
