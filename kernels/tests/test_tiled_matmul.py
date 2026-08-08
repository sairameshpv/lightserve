"""Correctness tests for kernels/tiled_matmul.py.

Reference is torch.matmul (cuBLAS on CUDA) -- the whole point of this kernel
is to compute the same thing cuBLAS does, just less efficiently (see
../matmul_ncu_report.md for by how much and why). Skipped on machines
without a CUDA GPU, same as kernels 1 and 2.
"""
import pytest
import torch

from kernels.tiled_matmul import matmul

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU (none available here)"
)

# (M, K, N) triples: a real LLM projection shape (QKVO-ish square, and
# FFN-shaped rectangular), plus small/non-block-multiple shapes to exercise
# the boundary-masking path on both the M/N (output) and K (reduction) axes.
SHAPES = [(1, 1, 1), (17, 33, 9), (129, 128, 130), (256, 256, 256), (512, 4096, 4096), (128, 4096, 14336)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@requires_cuda
@pytest.mark.parametrize("M,K,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_cublas_reference(M, K, N, dtype):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    actual = matmul(a, b)
    expected = torch.matmul(a, b)

    # GEMM error grows with K (more terms accumulated) -- fp16/bf16 need a
    # much looser tolerance than kernel 1/2's elementwise ops, and rtol is
    # scaled by sqrt(K) roughly in line with random-walk rounding-error
    # growth, rather than one fixed threshold for every K in the table.
    if dtype == torch.float32:
        atol, rtol = 1e-3, 1e-3
    else:
        # bf16/fp16 have a coarse ~7-bit mantissa (~1 part in 128 relative
        # step size), so small-K reductions can land one lucky/unlucky ULP
        # away from cuBLAS on a single element and still be entirely correct
        # bf16 arithmetic -- atol has a real floor here, not just rtol.
        atol, rtol = 1.5e-2, max(2e-2, 2e-2 * (K / 128) ** 0.5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_cuda
def test_matches_cublas_non_contiguous():
    # A transposed view has swapped strides -- exercises the kernel's
    # explicit stride arithmetic (both A and B use separate row/col
    # strides) instead of assuming row-major contiguous layout.
    torch.manual_seed(0)
    a_full = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    a = a_full.t()  # [128, 64], non-contiguous
    b = torch.randn(64, 96, device="cuda", dtype=torch.float32)

    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_rejects_inner_dim_mismatch():
    a = torch.randn(4, 8)
    b = torch.randn(9, 4)  # wrong: b's rows must match a's cols (8)
    with pytest.raises(AssertionError):
        matmul(a, b)


def test_rejects_dtype_mismatch():
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.randn(8, 4, dtype=torch.float16)
    with pytest.raises(AssertionError):
        matmul(a, b)