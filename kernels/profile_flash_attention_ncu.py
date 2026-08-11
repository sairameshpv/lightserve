"""Target script for Nsight Compute: runs ONE of (our tuned Triton kernel |
PyTorch's real FlashAttention-2 via SDPA forced onto the flash backend) for
a single bf16 attention shape, wrapping only the timed call in an NVTX range
so `ncu --nvtx --nvtx-include "PROFILE_TARGET/"` profiles exactly that
kernel launch. Same pattern as profile_matmul_ncu.py / profile_rmsnorm_ncu.py.

    ncu --nvtx --nvtx-include "PROFILE_TARGET/" --metrics <list> \
        --csv --log-file out.csv -- \
        python3 profile_flash_attention_ncu.py ours --N 4096 --causal
"""
import argparse

import torch
import torch.nn.functional as F

from flash_attention import flash_attention_forward


def sdpa_flash(q, k, v, causal):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except ImportError:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ours", "sdpa"])
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    q = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=torch.bfloat16)

    fn = (lambda: flash_attention_forward(q, k, v, causal=args.causal)) if args.mode == "ours" \
        else (lambda: sdpa_flash(q, k, v, args.causal))

    # Warm up outside the NVTX range: for "ours" this is what pays Triton's
    # autotune search cost across flash_attention.py's _CONFIGS, so none of
    # that one-time cost pollutes the profiled kernel below.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("PROFILE_TARGET")
    fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
