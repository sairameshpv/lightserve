"""Triton kernel 1: fused bias-add + ReLU.

Computes y = relu(x + bias) where x is [M, N] and bias is [N] (broadcast
across rows) — the pattern that normally follows a matmul/linear layer.
Fusing the two into one kernel avoids writing x+bias to HBM and reading it
back for the ReLU pass; see benchmark_fused_bias_relu.py for the measured
bandwidth win over doing it as two separate PyTorch ops.

This file is written as a hands-on walkthrough of Triton's block/tile
programming model, not just working code — read the comments in order.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Triton doesn't give each kernel "instance" (program) a single thread's
    # worth of work like raw CUDA does — it gives it a whole TILE (BLOCK_M x
    # BLOCK_N elements) to process with vectorized ops. The grid below is
    # launched as a 2D array of programs; each one asks "which tile am I?"
    # via program_id, one axis per grid dimension.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Turn "which tile" into the actual row/col indices this program owns.
    # tl.arange(0, BLOCK_M) is [0, 1, ..., BLOCK_M-1]; offsetting it by
    # pid_m*BLOCK_M slides that window to this program's slice of the M axis.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # M and N are runtime values, not necessarily multiples of BLOCK_M/N, so
    # the tile in the bottom-right corner of the grid can overhang the real
    # tensor. mask marks which (row, col) pairs in this tile are real vs
    # padding — every load/store below uses it so we never touch out-of-
    # bounds memory. [:, None] / [None, :] broadcast the 1D row/col masks
    # into a 2D BLOCK_M x BLOCK_N mask (outer AND).
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Pointer arithmetic: stride_xm/stride_xn are x's actual memory strides
    # (elements per step along each axis), so this works for both
    # contiguous and non-contiguous (e.g. transposed/sliced) inputs.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # bias is 1D over N only — one value per column, broadcast across every
    # row in the tile via [None, :], same as numpy/torch broadcasting rules.
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    y = tl.maximum(x + bias[None, :], 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y, mask=mask)


def fused_bias_relu(x: torch.Tensor, bias: torch.Tensor, block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """y = relu(x + bias). x: [M, N], bias: [N], same dtype, both on CUDA."""
    assert x.ndim == 2, f"x must be 2D, got shape {tuple(x.shape)}"
    assert bias.ndim == 1 and bias.shape[0] == x.shape[1], (
        f"bias shape {tuple(bias.shape)} must match x's last dim {x.shape[1]}"
    )
    assert x.is_cuda and bias.is_cuda, "Triton kernels need CUDA tensors (no CPU backend here)"
    assert x.dtype == bias.dtype, f"dtype mismatch: x={x.dtype} bias={bias.dtype}"

    M, N = x.shape
    out = torch.empty_like(x)

    # The grid: one program per (BLOCK_M x BLOCK_N) tile, ceil-divided so
    # the edge tiles (handled via the mask above) still get launched.
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _fused_bias_relu_kernel[grid](
        x, bias, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return out