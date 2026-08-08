"""Correctness tests for kernels/fused_rmsnorm_residual.py.

Reference is plain PyTorch eager, computed the same way LLaMA-family models
define RMSNorm: variance accumulated in fp32 regardless of input dtype, cast
back to the input dtype only at the very end. Skipped on machines without a
CUDA GPU, same as kernel 1's tests.
"""
import pytest
import torch

from kernels.fused_rmsnorm_residual import fused_add_rmsnorm

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU (none available here)"
)

# N values include real LLaMA-family hidden sizes (4096, 8192) plus
# non-power-of-2 widths (17, 300, 4097) to exercise BLOCK_N's masking path,
# and M=1/N=1 for degenerate single-row/single-column cases.
SHAPES = [(1, 1), (1, 17), (37, 300), (8, 4096), (4, 4097), (2, 8192)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def rmsnorm_residual_reference(x, residual, weight, eps):
    h = x + residual
    variance = h.float().pow(2).mean(-1, keepdim=True)
    h_normed = h.float() * torch.rsqrt(variance + eps)
    out = (h_normed * weight.float()).to(x.dtype)
    return out, h


@requires_cuda
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_pytorch_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    residual = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    actual_out, actual_residual = fused_add_rmsnorm(x, residual, weight, eps)
    expected_out, expected_residual = rmsnorm_residual_reference(x, residual, weight, eps)

    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(actual_residual, expected_residual, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_out, expected_out, atol=atol, rtol=rtol)


@requires_cuda
def test_output_is_unit_rms_before_weight():
    # Direct check on the math, independent of the reference implementation:
    # with weight == 1, mean(out**2) over each row should be ~1 by
    # definition of RMSNorm (that's what "RMS" normalization means).
    torch.manual_seed(0)
    M, N = 16, 512
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    residual = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.ones(N, device="cuda", dtype=torch.float32)

    out, _ = fused_add_rmsnorm(x, residual, weight, eps=1e-6)
    row_rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(row_rms, torch.ones(M, device="cuda"), atol=1e-3, rtol=1e-3)


@requires_cuda
def test_zero_row_stays_finite_via_eps():
    # A row of exact zeros has variance 0 -- without eps this is a 0/0. Make
    # sure eps actually keeps it finite instead of producing NaN.
    M, N = 4, 128
    x = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    residual = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)

    out, residual_out = fused_add_rmsnorm(x, residual, weight, eps=1e-6)
    assert torch.all(torch.isfinite(out))
    assert torch.all(residual_out == 0.0)


def test_rejects_shape_mismatch():
    x = torch.randn(4, 8)
    residual = torch.randn(4, 9)  # wrong: must match x's shape exactly
    weight = torch.randn(8)
    with pytest.raises(AssertionError):
        fused_add_rmsnorm(x, residual, weight)


def test_rejects_noncontiguous_input():
    # The kernel loads a whole row via tl.arange(0, BLOCK_N) at stride 1;
    # a transposed (non-contiguous-in-N) input would silently read garbage
    # if this weren't checked, so it must be rejected up front instead.
    x_full = torch.randn(8, 4)
    x = x_full.t()  # [4, 8], stride(1) != 1
    residual = torch.randn(4, 8)
    weight = torch.randn(8)
    with pytest.raises(AssertionError):
        fused_add_rmsnorm(x, residual, weight)