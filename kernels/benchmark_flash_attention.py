"""Benchmark: flash_attention_forward (Triton, kernel 4, v2/tuned) vs.
PyTorch's own FlashAttention-2 CUDA kernel, reached by forcing
`scaled_dot_product_attention` onto the FLASH_ATTENTION backend -- this
*is* "official FA-2" (or whatever version this torch build vendors), not an
approximation of it, so this is the same "ours vs. the real thing" shape as
kernel 3's ours-vs-cuBLAS benchmark.

bf16 throughout (the dtype _CONFIGS in flash_attention.py is tuned for --
tensor-core tl.dot, same reasoning as kernel 3's bf16-only benchmark).

    python3 benchmark_flash_attention.py

First call per shape pays Triton's autotune search cost (sweeping
flash_attention.py's _CONFIGS) -- warmup iterations happen before timing.
"""
import json
import statistics
import sys

import torch
import torch.nn.functional as F

from flash_attention import flash_attention_forward

DEVICE = "cuda"
DTYPE = torch.bfloat16
PEAK_BF16_TFLOPS = 362.0  # L40S dense bf16 spec, same constant as the roofline analysis and kernel 3

WARMUP = 10
ITERS = 30
REPEATS = 5

# (B, H, N, D): B=1 (matches the roofline doc's attention model, which is
# also implicitly batch=1), H=32/D=128 (LLaMA-3-8B's real head count/dim),
# N sweeping decode-adjacent up through long-context prefill.
SHAPES = [(1, 32, 512, 128), (1, 32, 1024, 128), (1, 32, 2048, 128), (1, 32, 4096, 128), (1, 32, 8192, 128)]


def sdpa_flash(q, k, v, causal):
    """Force PyTorch's real FlashAttention-2 CUDA kernel specifically --
    not "whatever backend SDPA picks" (which could silently fall back to
    the math or memory-efficient path and quietly stop being an FA2
    comparison). Raises if the flash backend can't run this call, rather
    than benchmark a different kernel under FA2's name."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except ImportError:
        # Older torch: same intent via the deprecated context-manager API.
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def attn_flops(B, H, N, D, causal):
    # QK^T and P@V, each 2*N*N*D FLOPs (multiply+add) per head -- same
    # formula as ../benchmarks/profiling/roofline/compute_roofline.py's
    # attention_score_prefill, extended with a batch factor and an explicit
    # causal halving (~half the (query, key) pairs are masked out and, with
    # the early-exit in flash_attention.py, never actually computed -- the
    # roofline doc's own model doesn't distinguish causal/non-causal FLOPs,
    # so this is a deliberate refinement, not a repeat of that formula).
    flops = 4 * B * H * N * N * D
    return flops // 2 if causal else flops


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

    print(f"{'shape (B,H,N,D)':<22}{'causal':>7}{'ours ms':>10}{'FA2 ms':>9}"
          f"{'ours TFLOPS':>13}{'FA2 TFLOPS':>12}{'% of FA2':>10}{'% of peak':>11}")
    print("-" * 96)

    for B, H, N, D in SHAPES:
        for causal in (False, True):
            torch.manual_seed(0)
            q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)

            actual = flash_attention_forward(q, k, v, causal=causal)
            expected = sdpa_flash(q, k, v, causal)
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

            ours_ms = time_block(lambda: flash_attention_forward(q, k, v, causal=causal))
            fa2_ms = time_block(lambda: sdpa_flash(q, k, v, causal))

            flops = attn_flops(B, H, N, D, causal)
            ours_tflops = flops / (ours_ms / 1000) / 1e12
            fa2_tflops = flops / (fa2_ms / 1000) / 1e12
            pct_of_fa2 = ours_tflops / fa2_tflops * 100
            pct_peak = ours_tflops / PEAK_BF16_TFLOPS * 100

            row = {
                "shape": f"{B}x{H}x{N}x{D}", "causal": causal,
                "ours_ms": round(ours_ms, 5), "fa2_ms": round(fa2_ms, 5),
                "ours_tflops": round(ours_tflops, 1), "fa2_tflops": round(fa2_tflops, 1),
                "pct_of_fa2": round(pct_of_fa2, 1), "pct_of_l40s_peak": round(pct_peak, 1),
            }
            results["rows"].append(row)
            print(f"{row['shape']:<22}{str(row['causal']):>7}{row['ours_ms']:>10.4f}{row['fa2_ms']:>9.4f}"
                  f"{row['ours_tflops']:>13.1f}{row['fa2_tflops']:>12.1f}"
                  f"{row['pct_of_fa2']:>9.1f}%{row['pct_of_l40s_peak']:>10.1f}%")

    print(f"\nAll {len(results['rows'])} (shape, causal) cases verified allclose against SDPA-flash before timing.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
