"""Correctness test for model/minimal_llama.py: the kernel-built forward
pass (matmul + fused_add_rmsnorm + flash_attention_forward) vs
`reference_llama_forward` (plain PyTorch, F.linear/SDPA/hand-written
RMSNorm), same random weights, same input ids.

Tolerance is looser than any single kernel's own test (e.g.
kernels/tests/test_flash_attention.py's bf16 2e-2/2e-2): this chains ~9
kernel calls per layer (2 matmuls' worth of norm, 3 QKV matmuls, 1 attention
call, 1 output matmul, 3 MLP matmuls) across n_layers layers, so per-kernel
bf16 rounding compounds rather than cancels -- a wider tolerance here is
measuring the same "order-of-summation differs from an fp32/eager
reference" story as every individual kernel test already does, just with
more of them chained, not a sign of an actual bug. fp32 is checked at a
tolerance close to individual-kernel tightness for the same reason it's
tight elsewhere: no precision-mode ambiguity to cause a legitimate gap.

Skipped on machines without a CUDA GPU, same as kernels 1-4.
"""
import math
from dataclasses import replace

import pytest
import torch

from model.minimal_llama import (
    LlamaConfig, TOY_CONFIG, _llama3_rope_scaling, init_weights, llama_forward, reference_llama_forward,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a CUDA GPU (none available here)"
)


@requires_cuda
@pytest.mark.parametrize("num_kv_heads", [None, 2])  # None=plain MHA, 2=GQA (n_heads=4)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,N", [(1, 8), (2, 37), (1, 64)])  # includes a non-block-multiple N (37)
def test_matches_reference(B, N, dtype, causal, num_kv_heads):
    torch.manual_seed(0)
    # fresh copy -- don't mutate the shared TOY_CONFIG instance
    config = replace(TOY_CONFIG, dtype=dtype, num_kv_heads=num_kv_heads)
    weights = init_weights(config, device="cuda", seed=0)
    input_ids = torch.randint(0, config.vocab_size, (B, N), device="cuda")

    logits = llama_forward(weights, config, input_ids, causal=causal)
    logits_ref = reference_llama_forward(weights, config, input_ids, causal=causal)

    assert logits.shape == (B, N, config.vocab_size)
    if dtype == torch.float32:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 8e-2, 8e-2
    torch.testing.assert_close(logits, logits_ref, atol=atol, rtol=rtol)


@requires_cuda
def test_matches_reference_single_layer_sanity():
    """A 1-layer model isolates whether a single sublayer pass (not
    compounded rounding across many layers) already agrees -- if this ever
    fails while the multi-layer test above also fails, the bug is in one
    sublayer, not accumulation.
    """
    torch.manual_seed(0)
    config = replace(TOY_CONFIG, n_layers=1, dtype=torch.float32)
    weights = init_weights(config, device="cuda", seed=0)
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

    logits = llama_forward(weights, config, input_ids)
    logits_ref = reference_llama_forward(weights, config, input_ids)
    torch.testing.assert_close(logits, logits_ref, atol=5e-3, rtol=5e-3)


def test_config_rejects_hidden_size_mismatch():
    # No CUDA needed -- pure config validation, runs on the CPU-only CI
    # runner (see .github/workflows/kernels-ci.yml).
    with pytest.raises(AssertionError):
        LlamaConfig(
            vocab_size=100, hidden_size=100, intermediate_size=64,
            n_layers=1, n_heads=4, head_dim=64,  # 4*64=256 != hidden_size=100
            max_seq_len=8,
        )


def test_config_rejects_unsupported_rope_scaling_type():
    with pytest.raises(AssertionError):
        LlamaConfig(
            vocab_size=100, hidden_size=256, intermediate_size=64,
            n_layers=1, n_heads=4, head_dim=64, max_seq_len=8,
            rope_scaling={"rope_type": "yarn"},  # only "llama3" is implemented
        )


def test_llama3_rope_scaling_matches_reference_formula():
    """_llama3_rope_scaling (vectorized torch.where) checked against an
    independent, per-frequency-index Python loop reimplementation of HF's
    llama3 rope_type formula -- same "second independent implementation of
    the same public math" pattern reference_llama_forward uses for the
    rest of the forward pass.
    """
    head_dim, rope_theta = 8, 500000.0
    rope_scaling = {
        "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192, "rope_type": "llama3",
    }
    half = head_dim // 2
    inv_freq = torch.tensor(
        [1.0 / (rope_theta ** (j / half)) for j in range(half)], dtype=torch.float64,
    )

    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    expected = []
    for f in inv_freq.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < high_freq_wavelen:
            expected.append(f)
        elif wavelen > low_freq_wavelen:
            expected.append(f / factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            expected.append(smooth * f / factor + (1 - smooth) * f)

    actual = _llama3_rope_scaling(inv_freq, rope_scaling)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-10, rtol=1e-10)
