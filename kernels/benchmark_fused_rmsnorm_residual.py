"""Benchmark: fused_add_rmsnorm (one Triton kernel) vs the naive PyTorch
eager sequence (add, then RMSNorm as separate mean/rsqrt/mul ops -- several
kernel launches, with `h = x + residual` written to HBM and read back for
the reduction instead of staying in registers).

Like kernel 1, this is a memory-traffic story, not a FLOPs story: the actual
arithmetic (one multiply-add per element for the residual, one square +
running sum + one rsqrt + two multiplies per element for the norm) is
trivial next to the HBM round-trips it takes to move x/residual/h/out.

    python3 benchmark_fused_rmsnorm_residual.py

Meant to run on a real CUDA GPU (Triton has no CPU backend).
"""
import json
import statistics
import sys

import torch

from fused_rmsnorm_residual import fused_add_rmsnorm

DEVICE = "cuda"
WARMUP = 15
ITERS = 100
REPEATS = 5

# (M, N) pairs spanning realistic LLM decode/prefill batch sizes (M) against
# real LLaMA-family hidden sizes (N = 4096, 8192).
SHAPES = [(1, 4096), (8, 4096), (128, 4096), (2048, 4096), (1, 8192), (128, 8192), (2048, 8192)]
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


def naive_add_rmsnorm(x, residual, weight, eps):
    h = x + residual
    variance = h.float().pow(2).mean(-1, keepdim=True)
    out = (h.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    return out, h


def bytes_moved(M, N, dtype_bytes):
    # fused: read x, read residual, read weight(~free, reused), write h,
    # write out -- 4 full [M,N] tensor touches (weight is O(N), negligible).
    fused = dtype_bytes * (4 * M * N)
    # naive (as PyTorch eager actually executes it):
    #   add:        read x, read residual, write h            -> 3*M*N
    #   h.float():  read h, write h_f32 (4 bytes/elem)         -> M*N*dtype_bytes + M*N*4
    #   pow+mean:   read h_f32, write variance ([M,1], ~free)  -> M*N*4
    #   rsqrt+mul+mul: read h_f32 (again) + write out          -> M*N*4 + M*N*dtype_bytes
    naive = (
        dtype_bytes * 3 * M * N
        + (dtype_bytes * M * N + 4 * M * N)
        + 4 * M * N
        + (4 * M * N + dtype_bytes * M * N)
    )
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
            residual = torch.randn(M, N, device=DEVICE, dtype=dtype)
            weight = torch.randn(N, device=DEVICE, dtype=dtype)
            eps = 1e-6

            actual_out, actual_res = fused_add_rmsnorm(x, residual, weight, eps)
            expected_out, expected_res = naive_add_rmsnorm(x, residual, weight, eps)
            atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-2, 2e-2)
            torch.testing.assert_close(actual_res, expected_res, atol=atol, rtol=rtol)
            torch.testing.assert_close(actual_out, expected_out, atol=atol, rtol=rtol)

            fused_ms = time_block(lambda: fused_add_rmsnorm(x, residual, weight, eps))
            naive_ms = time_block(lambda: naive_add_rmsnorm(x, residual, weight, eps))

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
          f"against the PyTorch-eager add+RMSNorm reference before timing.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
