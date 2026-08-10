"""Correctness tests for kernels/flash_attention.py.

Reference is torch.nn.functional.scaled_dot_product_attention (SDPA) with
its default scale (1/sqrt(D)), matching this kernel's default. v1 is fp32
only -- see the module docstring for why (correctness-first, no tensor-core
dtype path yet). Skipped on machines without a CUDA GPU, same as kernels
1-3.
"""
import pytest
import torch
import torch.nn.functional as F

from kernels.flash_attention import flash_attention_forward

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU (none available here)"
)

# (B, H, N, D) shapes: a couple of LLM-real head dims (64, 128) at
# block-aligned N, plus small/non-block-multiple N to exercise the
# boundary-masking path (BLOCK_M/BLOCK_N default to 64), plus a
# non-power-of-2 D to exercise BLOCK_D's masking path independently of N's.
SHAPES = [
    (1, 1, 1, 8),
    (1, 1, 17, 32),
    (2, 2, 64, 64),
    (1, 4, 130, 64),
    (2, 8, 256, 128),
    (1, 2, 100, 96),
]


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_matches_sdpa_reference(B, H, N, D, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    actual = flash_attention_forward(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # fp32, IEEE tl.dot (not TF32) on both matmuls -- same tight tolerance
    # kernel 3 uses for its fp32 path, since we're matching the same
    # precision mode SDPA computes in, not approximating it.
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@requires_cuda
def test_matches_sdpa_non_contiguous():
    # A transposed-heads view has non-default strides on the N/D axes --
    # exercises the kernel's explicit stride arithmetic instead of assuming
    # a contiguous [B, H, N, D] layout.
    torch.manual_seed(0)
    q_full = torch.randn(1, 64, 4, 32, device="cuda", dtype=torch.float32)
    q = q_full.transpose(1, 2)  # [1, 4, 64, 32], non-contiguous
    k = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float32)

    actual = flash_attention_forward(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_rejects_shape_mismatch():
    q = torch.randn(1, 2, 16, 32)
    k = torch.randn(1, 2, 16, 32)
    v = torch.randn(1, 2, 8, 32)  # wrong: N must match q/k
    with pytest.raises(AssertionError):
        flash_attention_forward(q, k, v)


def test_rejects_dtype_mismatch():
    q = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 32, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    with pytest.raises(AssertionError):
        flash_attention_forward(q, k, v)