"""Target script for Nsight Compute: runs ONE of (fused kernel | naive
PyTorch eager) add+RMSNorm, wrapping only the timed call in an NVTX range so
`ncu --nvtx --nvtx-include "PROFILE_TARGET/"` profiles exactly those kernels
and nothing from CUDA-context/cuBLAS-handle initialization or the untimed
warmup calls before it.

    ncu --nvtx --nvtx-include "PROFILE_TARGET/" \
        --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,... \
        --csv --log-file out.csv -- \
        python3 profile_rmsnorm_ncu.py fused --M 2048 --N 4096 --dtype bfloat16
"""
import argparse

import torch

from fused_rmsnorm_residual import fused_add_rmsnorm


def naive_add_rmsnorm(x, residual, weight, eps):
    h = x + residual
    variance = h.float().pow(2).mean(-1, keepdim=True)
    out = (h.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    return out, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["fused", "naive"])
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, device="cuda", dtype=dtype)
    residual = torch.randn(args.M, args.N, device="cuda", dtype=dtype)
    weight = torch.randn(args.N, device="cuda", dtype=dtype)
    eps = 1e-6

    fn = (lambda: fused_add_rmsnorm(x, residual, weight, eps)) if args.mode == "fused" \
        else (lambda: naive_add_rmsnorm(x, residual, weight, eps))

    # Warm up OUTSIDE the NVTX range: JIT-compiles the Triton kernel / warms
    # cuBLAS-adjacent lazy dispatch, so none of that one-time cost pollutes
    # the profiled kernels below.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("PROFILE_TARGET")
    fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()