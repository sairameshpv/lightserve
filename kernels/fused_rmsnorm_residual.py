"""Triton kernel 2: fused residual-add + RMSNorm.

Computes, for x/residual: [M, N], weight: [N]:

    h   = x + residual                       # new residual, carried forward
    out = h / sqrt(mean(h**2, dim=-1) + eps) * weight

This is the exact pattern at every pre-norm transformer decoder layer
boundary (LLaMA/Mistral/etc.): `x` is the previous sublayer's output,
`residual` is the skip-connection value, `h` becomes the residual fed into
the *next* sublayer, and `out` is what that next sublayer (attention or FFN)
actually consumes. Fusing add+norm into one kernel means `h` never round-
trips through HBM as a separate tensor between two kernel launches -- PyTorch
eager computes it as its own op (`x + residual`), writes it out, and then a
*second* kernel reads it back in for the mean-of-squares reduction.

Unlike kernel 1 (fused_bias_relu.py, a purely elementwise tile-parallel op),
this kernel needs a per-row REDUCTION (mean of squares across all of N)
before it can produce a single output element -- so the tiling strategy is
different: one Triton program per ROW, each program loading and reducing
its *entire* row in one shot (BLOCK_N >= N, masked). That only works because
N (the model's hidden size) is small enough to fit a whole row in registers;
a kernel over a reduction axis too large for that would need a multi-pass
(split-K style) reduction instead.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr, residual_out_ptr,
    N,
    stride_xm, stride_rm, stride_om, stride_rom,
    eps,
    BLOCK_N: tl.constexpr,
):
    # One program per row -- program_id directly IS the row index, no 2D
    # tiling needed since every program consumes the whole row at once.
    row = tl.program_id(axis=0)

    # BLOCK_N is the next power of 2 >= N (Triton requires arange's length to
    # be a compile-time power of 2), so it can overhang the real row width --
    # same masking discipline as kernel 1, just 1D instead of 2D here.
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_row = tl.load(x_ptr + row * stride_xm + offs_n, mask=mask, other=0.0)
    residual_row = tl.load(residual_ptr + row * stride_rm + offs_n, mask=mask, other=0.0)

    # This IS the fusion: h is computed once, in registers, and used twice
    # below (stored as the new residual, AND reduced for the norm) without
    # ever touching HBM in between.
    h = x_row + residual_row
    tl.store(residual_out_ptr + row * stride_rom + offs_n, h, mask=mask)

    # Variance (mean of squares, RMSNorm has no mean-subtraction unlike
    # LayerNorm) is accumulated in fp32 regardless of input dtype -- bf16/fp16
    # squared-and-summed over thousands of elements loses real precision
    # otherwise. Masked-out lanes hold 0.0 (from `other=0.0` above), so they
    # contribute nothing to the sum; dividing by the true N (not BLOCK_N)
    # keeps the mean correct.
    h_f32 = h.to(tl.float32)
    variance = tl.sum(h_f32 * h_f32, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    weight = tl.load(weight_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    out = (h_f32 * rrms * weight).to(h.dtype)
    tl.store(out_ptr + row * stride_om + offs_n, out, mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
):
    """h = x + residual; returns (rmsnorm(h) * weight, h).

    x, residual: [M, N] same dtype/device. weight: [N]. h is returned because
    the caller needs it as the residual for the next sublayer -- this is the
    same two-output contract vLLM/TRT-LLM's fused add+RMSNorm kernels use.
    """
    assert x.shape == residual.shape and x.ndim == 2, (
        f"x {tuple(x.shape)} and residual {tuple(residual.shape)} must both be 2D and equal-shaped"
    )
    assert weight.ndim == 1 and weight.shape[0] == x.shape[1], (
        f"weight shape {tuple(weight.shape)} must match x's last dim {x.shape[1]}"
    )
    assert x.is_cuda and residual.is_cuda and weight.is_cuda, "Triton kernels need CUDA tensors"
    assert x.dtype == residual.dtype == weight.dtype, (
        f"dtype mismatch: x={x.dtype} residual={residual.dtype} weight={weight.dtype}"
    )
    # The kernel loads a full row into one program via tl.arange(0, BLOCK_N),
    # which requires contiguous rows (stride 1 along N) -- unlike kernel 1,
    # this one doesn't take a stride_xn/stride_rn parameter.
    assert x.stride(1) == 1 and residual.stride(1) == 1, (
        "x and residual must be contiguous along the last dim (row-major, no transposed views)"
    )

    M, N = x.shape
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N)
    # More columns per row -> more parallel work per program -> more warps
    # can usefully split it. Capped at 16, Triton's usual ceiling per CTA.
    num_warps = min(max(BLOCK_N // 256, 1), 16)

    grid = (M,)
    _fused_add_rmsnorm_kernel[grid](
        x, residual, weight, out, residual_out,
        N,
        x.stride(0), residual.stride(0), out.stride(0), residual_out.stride(0),
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out, residual_out
