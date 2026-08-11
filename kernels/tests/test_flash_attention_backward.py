"""Gradient correctness tests for kernels/flash_attention_backward.py.

Two independent checks, same "don't trust a single verification method"
spirit as the forward kernel's own test file:

  1. `test_gradcheck` -- torch.autograd.gradcheck: numerical
     (finite-difference) vs analytical gradient agreement. This is the
     standard way to validate a custom torch.autograd.Function's backward()
     is the true Jacobian of its forward(), and it's independent of any
     reference attention implementation -- it only trusts calculus, not
     PyTorch's own SDPA.
  2. `test_matches_sdpa_autograd` -- analytical-vs-analytical: our dQ/dK/dV
     vs PyTorch's own autograd through
     torch.nn.functional.scaled_dot_product_attention, same "match the real
     thing" shape as flash_attention.py's forward-pass SDPA comparison, now
     extended to gradients.

gradcheck runs at float32, not the textbook float64: tl.dot (the
tensor-core matmul primitive both the forward and backward kernels are
built on) isn't exercised at float64 anywhere in this repo -- none of
kernels 1-4's tests use it either -- so this avoids depending on a code
path this project has never actually run on real hardware. float32
finite-differencing needs a coarser `eps` and looser `atol`/`rtol` than
float64's defaults to keep round-off from swamping the finite-difference
signal; that's what's tuned below, not a weaker check in spirit.

Skipped on machines without a CUDA GPU, same as kernels 1-4.
"""
import pytest
import torch
import torch.nn.functional as F

from kernels.flash_attention_backward import flash_attention

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU (none available here)"
)

# Deliberately tiny: gradcheck perturbs and re-evaluates the forward pass
# roughly 2x per input element for central differences (2 * B*H*N*D calls
# total across q/k/v), so this is already O(minutes) on a real GPU at these
# sizes -- LLM-real shapes are for the (much cheaper, single forward+backward
# pass) SDPA comparison test below instead.
GRADCHECK_SHAPES = [
    (1, 1, 4, 8),
    (1, 2, 9, 16),  # non-block-multiple N (BLOCK_M=BLOCK_N=32) -- exercises the backward path's masking too
]


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", GRADCHECK_SHAPES)
def test_gradcheck(B, H, N, D, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda q, k, v: flash_attention(q, k, v, causal=causal),
        (q, k, v),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
    )


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,H,N,D", [(1, 1, 17, 32), (2, 2, 64, 64), (1, 4, 130, 64), (2, 8, 256, 128)])
def test_matches_sdpa_autograd(B, H, N, D, dtype, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    do = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    out = flash_attention(q, k, v, causal=causal)
    out.backward(do)

    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=causal)
    out_ref.backward(do)

    if dtype == torch.float32:
        # IEEE tl.dot on both matmuls in both forward and backward -- same
        # tight tolerance the forward test uses for fp32, matching (not
        # approximating) SDPA's own precision.
        atol, rtol = 1e-3, 1e-3
    else:
        # bf16: coarse ~7-bit mantissa, plus the backward path chains
        # several more matmuls (dV, dP, dK, dQ) than the forward's one --
        # same order-of-summation-differs-from-reference story as the
        # forward test's bf16 tolerance, widened a bit further for the
        # extra matmul depth.
        atol, rtol = 3e-2, 3e-2

    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@requires_cuda
def test_forward_matches_forward_only_kernel():
    """flash_attention()'s forward output must match flash_attention.py's
    forward-only flash_attention_forward() -- confirms the LSE-emitting
    forward kernel added for backward (a separate, non-autotuned copy of
    the same loop, see flash_attention_backward.py's docstring) didn't
    silently change the forward math itself.
    """
    from kernels.flash_attention import flash_attention_forward

    torch.manual_seed(0)
    q = torch.randn(2, 4, 130, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 130, 64, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 4, 130, 64, device="cuda", dtype=torch.float32)

    out_lse = flash_attention(q, k, v, causal=True)
    out_fwd_only = flash_attention_forward(q, k, v, causal=True)
    torch.testing.assert_close(out_lse, out_fwd_only, atol=1e-3, rtol=1e-3)


def test_rejects_shape_mismatch():
    q = torch.randn(1, 2, 16, 32)
    k = torch.randn(1, 2, 16, 32)
    v = torch.randn(1, 2, 8, 32)  # wrong: N must match q/k
    with pytest.raises(AssertionError):
        flash_attention(q, k, v)


def test_rejects_dtype_mismatch():
    q = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 32, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    with pytest.raises(AssertionError):
        flash_attention(q, k, v)