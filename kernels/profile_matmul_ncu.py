"""Target script for Nsight Compute: runs ONE of (our tiled kernel | cuBLAS
via torch.matmul) for a single square bf16 GEMM, wrapping only the timed
call in an NVTX range so `ncu --nvtx --nvtx-include "PROFILE_TARGET/"`
profiles exactly that kernel launch. Same pattern as profile_rmsnorm_ncu.py.

    ncu --nvtx --nvtx-include "PROFILE_TARGET/" --metrics <list> \
        --csv --log-file out.csv -- \
        python3 profile_matmul_ncu.py ours --size 4096
"""
import argparse

import torch

from tiled_matmul import matmul


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ours", "cublas"])
    p.add_argument("--size", type=int, default=4096)
    args = p.parse_args()

    torch.manual_seed(0)
    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)

    fn = (lambda: matmul(a, b)) if args.mode == "ours" else (lambda: torch.matmul(a, b))

    # Warm up outside the NVTX range: for "ours" this is what pays Triton's
    # autotune search cost across tiled_matmul.py's configs, so none of that
    # one-time cost pollutes the profiled kernel below.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("PROFILE_TARGET")
    fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()