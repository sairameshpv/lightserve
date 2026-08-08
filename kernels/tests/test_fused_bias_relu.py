"""Correctness tests for kernels/fused_bias_relu.py.

Reference is plain PyTorch (torch.relu(x + bias)) — the whole point of the
kernel is to be numerically equivalent to that while doing one HBM
round-trip instead of two. Skipped entirely on machines without a CUDA GPU
(Triton has no real CPU backend), so this passes-by-skipping on standard
GitHub-hosted CI runners and only truly validates on GPU hardware.
"""
import pytest
import torch

from kernels.fused_bias_relu import fused_bias_relu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU (none available here)"
)

# Includes shapes that are NOT multiples of the default 64x64 block size,
# to exercise the boundary-masking path (M=1 and N=1 also cover degenerate
# single-row/single-column tiles).
SHAPES = [(1, 1), (1, 130), (37, 17), (64, 64), (128, 128), (200, 300), (4096, 4096)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@requires_cuda
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_pytorch_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)

    actual = fused_bias_relu(x, bias)
    expected = torch.relu(x + bias)

    # bf16/fp16 accumulate more rounding error than fp32; loosen tolerances
    # accordingly rather than using one blanket threshold for every dtype.
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_cuda
def test_matches_pytorch_reference_non_contiguous():
    # A transposed view has swapped strides, not the default row-major
    # layout — exercises the kernel's explicit stride arithmetic instead of
    # assuming contiguous memory.
    torch.manual_seed(0)
    x_full = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x = x_full.t()  # [128, 64], non-contiguous
    bias = torch.randn(64, device="cuda", dtype=torch.float32)

    actual = fused_bias_relu(x, bias)
    expected = torch.relu(x + bias)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_all_negative_inputs_zero_out():
    # Sanity check on the ReLU half specifically: everything below -bias
    # should clip to exactly 0, not just "close to" 0.
    x = torch.full((32, 32), -100.0, device="cuda", dtype=torch.float32)
    bias = torch.zeros(32, device="cuda", dtype=torch.float32)
    actual = fused_bias_relu(x, bias)
    assert torch.all(actual == 0.0)


def test_rejects_shape_mismatch():
    # Shape validation runs before any CUDA/Triton call, so this is worth
    # checking even without a GPU present.
    x = torch.randn(4, 8)
    bias = torch.randn(4)  # wrong length: should match x.shape[1]==8
    with pytest.raises(AssertionError):
        fused_bias_relu(x, bias)