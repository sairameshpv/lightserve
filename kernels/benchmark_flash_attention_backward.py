"""Benchmark: flash_attention_backward's backward pass (Triton, kernel 4b,
recomputation strategy) vs. PyTorch's own FlashAttention-2 backward, reached
the same way kernel 4's forward benchmark reaches FA2 forward -- forcing
`scaled_dot_product_attention` onto the FLASH_ATTENTION backend, then
calling `.backward()` through it. Same "ours vs. the real thing" shape as
benchmark_flash_attention.py and kernel 3's ours-vs-cuBLAS benchmark.

Unlike kernel 4's forward (v2: autotuned, causal-early-exited, the one
actually competitive with FA2 -- see kernels/README.md), this backward
kernel is still "v1" by its own docstring: fixed tile size (16x16 or 32x32
depending on dtype/head-dim, see `_pick_block_sizes`), no
`@triton.autotune`. This benchmark is expected to show a real gap, the same
way kernel 4's own forward v1 (never benchmarked for speed, correctness-only
by design) would have -- that's the point of running it, not a surprise
result.

bf16 throughout, same reasoning as kernel 4's forward benchmark. Shapes
stop at N=4096 (forward's benchmark goes to 8192): backward does ~2.5x
forward's FLOPs (5 matmuls -- recompute S, dV, dP, dK, dQ -- vs forward's 2,
same accounting the FlashAttention-2 paper itself uses) through a
non-autotuned kernel, so N=8192 would cost real GPU time for a data point
this file already expects to be far from parity.

    python3 benchmark_flash_attention_backward.py
"""
import json
import statistics
import sys

import torch
import torch.nn.functional as F

from flash_attention_backward import flash_attention

DEVICE = "cuda"
DTYPE = torch.bfloat16
PEAK_BF16_TFLOPS = 362.0  # L40S dense bf16 spec, same constant as kernel 3/4's benchmarks

WARMUP = 5
ITERS = 10
REPEATS = 3

SHAPES = [(1, 32, 512, 128), (1, 32, 1024, 128), (1, 32, 2048, 128), (1, 32, 4096, 128)]


def sdpa_flash(q, k, v, causal):
    """Same FA2-forcing helper as benchmark_flash_attention.py -- see its
    docstring for why this, not "whatever backend SDPA picks", is the
    comparison target."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except ImportError:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def attn_fwd_flops(B, H, N, D, causal):
    # Same formula as benchmark_flash_attention.py's attn_flops.
    flops = 4 * B * H * N * N * D
    return flops // 2 if causal else flops


def attn_bwd_flops(B, H, N, D, causal):
    # 5 matmul-shaped ops (recompute S, dV=P^T@dO, dP=dO@V^T, dK=dS^T@Q,
    # dQ=dS@K) vs forward's 2 (QK^T, P@V) -- see module docstring.
    return attn_fwd_flops(B, H, N, D, causal) * 5 // 2


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


def make_backward_step(fwd_fn, q, k, v, do, causal):
    """Builds the autograd graph ONCE (outside the timed region) via a
    single forward call, then returns a closure that only re-runs
    `.backward(retain_graph=True)` -- isolates backward-pass time from
    forward-pass time, which time_block's own warmup+timed loop then
    exercises repeatedly on the same graph.
    """
    q = q.detach().clone().requires_grad_()
    k = k.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    out = fwd_fn(q, k, v, causal=causal)

    def step():
        q.grad = None
        k.grad = None
        v.grad = None
        out.backward(do, retain_graph=True)

    return step


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
            q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
            k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
            v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
            do = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)

            # Verify gradients agree before timing either -- same
            # "allclose-verified before every timed shape" discipline as
            # benchmark_flash_attention.py, extended to grads (same
            # tolerance kernels/tests/test_flash_attention_backward.py's
            # bf16 SDPA-autograd comparison uses).
            q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
            flash_attention(q, k, v, causal=causal).backward(do)
            sdpa_flash(q_ref, k_ref, v_ref, causal).backward(do)
            torch.testing.assert_close(q.grad, q_ref.grad, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(k.grad, k_ref.grad, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(v.grad, v_ref.grad, atol=3e-2, rtol=3e-2)

            ours_step = make_backward_step(flash_attention, q, k, v, do, causal)
            fa2_step = make_backward_step(sdpa_flash, q, k, v, do, causal)
            ours_ms = time_block(ours_step)
            fa2_ms = time_block(fa2_step)

            flops = attn_bwd_flops(B, H, N, D, causal)
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

    print(f"\nAll {len(results['rows'])} (shape, causal) cases verified allclose against SDPA-flash grads before timing.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
