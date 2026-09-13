"""A minimal LLaMA-shaped decoder-only transformer, forward pass built from
this repo's own Triton kernels (kernels/tiled_matmul.py, kernels/
fused_rmsnorm_residual.py, kernels/flash_attention.py) instead of
torch.nn.Linear / F.rms_norm / F.scaled_dot_product_attention.

Scope, on purpose ("minimal"):

  - kernels/flash_attention.py's kernel itself is still MHA-only
    (requires q.shape == k.shape == v.shape, same H) -- LLaMA-3's real
    grouped-query attention (8 KV heads vs 32 Q heads) is expressed on top
    of it by `repeat_kv`-broadcasting K/V up to n_heads before every
    attention call (see LlamaConfig.num_kv_heads, model_runner.py's
    `_attention`), not by changing the kernel.
  - RoPE and the MLP's SiLU-gate-multiply are plain PyTorch ops, not fused
    Triton kernels. Both are pure elementwise/broadcast math with no
    reduction and no reuse-across-threads to exploit (unlike kernel 1's
    bias+ReLU, which fuses two HBM round-trips into one) -- there's a real
    fused-RoPE or fused-SwiGLU kernel to write eventually, just not the
    point of this file, which is *integrating* the kernels that already
    exist (matmul, add+RMSNorm, FlashAttention) into a real model shape.
  - Linear-layer weights are stored [in_features, out_features] (the
    transpose of nn.Linear's usual [out, in]) purely so `matmul(x, w)`
    (kernels/tiled_matmul.py's `C = A @ B`, no transpose argument) can be
    called directly -- no separate transpose kernel or op needed.
  - `init_weights` below is random, not a real LLaMA-3 checkpoint -- kept
    for the systems/integration tests and benchmarks in this file's own
    test/benchmark scripts, where the point is whether the forward pass
    composes correctly and is fast, not output quality.
    `reference_llama_forward` (pure PyTorch, same weights) is what
    correctness is checked against -- see model/tests/test_minimal_llama.py.
    `model/hf_loader.py` loads a real checkpoint into these same
    LlamaConfig/LlamaWeights dataclasses when real weights are wanted (see
    model/tests/test_hf_loader.py).

LLAMA3_8B_SHAPE below uses LLaMA-3-8B-Instruct's real per-layer dimensions
(hidden=4096, n_heads=32, head_dim=128, intermediate=14336, vocab=128256 --
the same model benchmarks/run_baseline.py and kernels/benchmark_flash_
attention.py already target) with `n_layers` overridable -- the real model
has 32; running all 32 isn't necessary to demonstrate the integration or
the CUDA-graph mechanics in model/cuda_graph_decode.py, so that script
defaults to a handful of layers and lets you scale up.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from kernels.flash_attention import flash_attention_forward
from kernels.fused_rmsnorm_residual import fused_add_rmsnorm
from kernels.tiled_matmul import matmul


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    n_layers: int
    n_heads: int
    head_dim: int
    max_seq_len: int
    # None -> plain MHA (num_kv_heads == n_heads). Set it lower for
    # grouped-query attention (LLaMA-3-8B: 8 KV heads vs 32 Q heads); the
    # kernel is still MHA-only, so model_runner.py/minimal_llama.py
    # repeat_kv-broadcast K/V up to n_heads before every attention call.
    num_kv_heads: int = None
    rope_theta: float = 500000.0  # LLaMA-3's real value (LLaMA-1/2 used 10000)
    # None -> plain RoPE. Llama-3.1/3.2's config.json sets this to a dict
    # (HF's "llama3" rope_type -- NTK-aware frequency scaling for
    # long-context models); see precompute_rope. Only "llama3" is
    # implemented -- the other HF rope_types (linear/dynamic/yarn/
    # longrope) use different formulas this repo doesn't have.
    rope_scaling: dict = None
    rms_eps: float = 1e-5
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        assert self.hidden_size == self.n_heads * self.head_dim, (
            f"hidden_size ({self.hidden_size}) must equal n_heads*head_dim "
            f"({self.n_heads}*{self.head_dim}={self.n_heads * self.head_dim})"
        )
        if self.num_kv_heads is None:
            self.num_kv_heads = self.n_heads
        assert self.n_heads % self.num_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be a multiple of num_kv_heads "
            f"({self.num_kv_heads})"
        )
        if self.rope_scaling is not None:
            assert self.rope_scaling.get("rope_type") == "llama3", (
                f"only rope_scaling rope_type 'llama3' is implemented, got "
                f"{self.rope_scaling.get('rope_type')!r}"
            )


# Small, fast -- for correctness tests and quick smoke runs.
TOY_CONFIG = LlamaConfig(
    vocab_size=256, hidden_size=256, intermediate_size=688,
    n_layers=2, n_heads=4, head_dim=64, max_seq_len=64,
)


def llama3_8b_shape(n_layers: int = 4, max_seq_len: int = 128) -> LlamaConfig:
    """LLaMA-3-8B-Instruct's real per-layer shape, `n_layers` truncated from
    the real 32 (see module docstring) -- for model/cuda_graph_decode.py.
    """
    return LlamaConfig(
        vocab_size=128256, hidden_size=4096, intermediate_size=14336,
        n_layers=n_layers, n_heads=32, head_dim=128, max_seq_len=max_seq_len,
    )


@dataclass
class LayerWeights:
    input_layernorm_weight: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass
class LlamaWeights:
    embed_tokens: torch.Tensor
    layers: list = field(default_factory=list)
    norm_weight: torch.Tensor = None
    lm_head: torch.Tensor = None


def init_weights(config: LlamaConfig, device: str = "cuda", seed: int = 0) -> LlamaWeights:
    """Random init (std=0.02, the usual transformer default) -- not a real
    checkpoint, see module docstring.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * 0.02).to(config.dtype)

    def ones(*shape):
        return torch.ones(*shape, device=device, dtype=config.dtype)

    H, I, V = config.hidden_size, config.intermediate_size, config.vocab_size
    qkv_dim = config.n_heads * config.head_dim
    kv_dim = config.num_kv_heads * config.head_dim  # == qkv_dim under plain MHA

    layers = []
    for _ in range(config.n_layers):
        layers.append(LayerWeights(
            input_layernorm_weight=ones(H),
            q_proj=randn(H, qkv_dim),
            k_proj=randn(H, kv_dim),
            v_proj=randn(H, kv_dim),
            o_proj=randn(qkv_dim, H),
            post_attention_layernorm_weight=ones(H),
            gate_proj=randn(H, I),
            up_proj=randn(H, I),
            down_proj=randn(I, H),
        ))

    return LlamaWeights(
        embed_tokens=randn(V, H),
        layers=layers,
        norm_weight=ones(H),
        lm_head=randn(H, V),
    )


def _llama3_rope_scaling(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
    """HF's Llama-3.1/3.2 NTK-aware RoPE frequency scaling (rope_type
    "llama3", see LlamaConfig.rope_scaling): short wavelengths (high
    frequencies) are left alone, long wavelengths (low frequencies) are
    divided by `factor`, and the band in between is smoothly interpolated
    -- lets a model trained at original_max_position_embeddings extrapolate
    to a much longer max_position_embeddings without retraining.
    """
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = smooth * scaled / factor + (1 - smooth) * scaled
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def precompute_rope(config: LlamaConfig, device: str):
    """cos/sin lookup tables, [max_seq_len, head_dim] each -- a fixed buffer
    sliced identically on every call (same [0:N] range every decode step in
    this file's "always run the full max_seq_len buffer" design, see
    cuda_graph_decode.py), so this needs no dynamic control flow and is
    trivially CUDA-graph-safe.
    """
    half = config.head_dim // 2
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    if config.rope_scaling is not None:
        inv_freq = _llama3_rope_scaling(inv_freq, config.rope_scaling)
    positions = torch.arange(config.max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(config.dtype)  # [max_seq_len, head_dim]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(config.dtype)
    return cos, sin


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin, n):
    """q, k: [B, n_heads, N, head_dim]. cos/sin: [max_seq_len, head_dim],
    sliced to this call's N and broadcast over batch/heads.
    """
    cos_n = cos[:n].view(1, 1, n, -1)
    sin_n = sin[:n].view(1, 1, n, -1)
    q_rot = q * cos_n + _rotate_half(q) * sin_n
    k_rot = k * cos_n + _rotate_half(k) * sin_n
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Broadcasts the head dim (always x's dim 1, e.g. this file's own
    [B, num_kv_heads, N, head_dim] or model_runner.py::_attention's
    [seq_len, num_kv_heads, head_dim]) from num_kv_heads up to
    num_kv_heads*n_rep. Each KV head is repeated n_rep times contiguously
    (HF's repeat_kv layout: heads [0,0,1,1,...] for n_rep=2, not interleaved
    [0,1,0,1,...]) so query head i attends to KV head i // n_rep. No-op
    (returns x) when n_rep == 1, i.e. plain MHA.
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _linear(x_2d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x_2d: [M, in], w: [in, out] -- see module docstring on why weights
    are stored transposed from nn.Linear's convention.
    """
    return matmul(x_2d, w)


def llama_forward(weights: LlamaWeights, config: LlamaConfig, input_ids: torch.Tensor,
                   causal: bool = True) -> torch.Tensor:
    """input_ids: [B, N] int64 token ids. Returns logits [B, N, vocab_size].

    Every Linear is kernels/tiled_matmul.py's `matmul`, every
    residual-add+norm is kernels/fused_rmsnorm_residual.py's
    `fused_add_rmsnorm`, every attention call is kernels/flash_attention.py's
    `flash_attention_forward` (causal, bf16-tuned) -- RoPE and the MLP's
    SiLU-gate are plain PyTorch (see module docstring).
    """
    B, N = input_ids.shape
    H = config.hidden_size
    cos, sin = precompute_rope(config, input_ids.device)

    x = F.embedding(input_ids, weights.embed_tokens)  # [B, N, H]
    residual = x
    # First sublayer's "x" input to fused_add_rmsnorm is the embedding
    # itself; residual starts equal to it so the first add is x+x's own
    # copy... instead, feed a zero x so the first norm is just norm(residual)
    # with no double-count. Cheaper than a special-cased "first layer" path.
    x = torch.zeros_like(x)

    for layer in weights.layers:
        normed, residual = fused_add_rmsnorm(
            x.reshape(B * N, H), residual.reshape(B * N, H),
            layer.input_layernorm_weight, eps=config.rms_eps,
        )
        normed = normed.reshape(B, N, H)
        residual = residual.reshape(B, N, H)
        normed_2d = normed.reshape(B * N, H)

        q = _linear(normed_2d, layer.q_proj).reshape(B, N, config.n_heads, config.head_dim).transpose(1, 2)
        k = _linear(normed_2d, layer.k_proj).reshape(B, N, config.num_kv_heads, config.head_dim).transpose(1, 2)
        v = _linear(normed_2d, layer.v_proj).reshape(B, N, config.num_kv_heads, config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, N)
        # GQA: kernel needs q/k/v head counts to match (see module docstring),
        # so broadcast K/V's fewer heads up to n_heads here. No-op under MHA.
        n_rep = config.n_heads // config.num_kv_heads
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)

        attn_out = flash_attention_forward(q, k, v, causal=causal)  # [B, n_heads, N, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B * N, config.n_heads * config.head_dim)
        x = _linear(attn_out, layer.o_proj).reshape(B, N, H)

        normed2, residual = fused_add_rmsnorm(
            x.reshape(B * N, H), residual.reshape(B * N, H),
            layer.post_attention_layernorm_weight, eps=config.rms_eps,
        )
        residual = residual.reshape(B, N, H)

        gate = _linear(normed2, layer.gate_proj)
        up = _linear(normed2, layer.up_proj)
        mlp_hidden = F.silu(gate) * up
        x = _linear(mlp_hidden, layer.down_proj).reshape(B, N, H)

    # Final norm: `residual` here is still only attention-updated -- each
    # layer's MLP output only gets folded into it via the *next* layer's
    # fused_add_rmsnorm call (its `h = x + residual` add), so the very last
    # layer's MLP output (this loop's final `x`) has no "next layer" to do
    # that fold for it. Feed it in explicitly here instead of the zero this
    # used to pass -- passing zero silently dropped the last layer's MLP
    # sublayer entirely (caught by model/tests/test_minimal_llama.py
    # against the plain-PyTorch reference, which does `residual = residual
    # + x` after every MLP unconditionally).
    final_normed, _ = fused_add_rmsnorm(
        x.reshape(B * N, H), residual.reshape(B * N, H),
        weights.norm_weight, eps=config.rms_eps,
    )
    logits = _linear(final_normed, weights.lm_head).reshape(B, N, config.vocab_size)
    return logits


def reference_llama_forward(weights: LlamaWeights, config: LlamaConfig, input_ids: torch.Tensor,
                             causal: bool = True) -> torch.Tensor:
    """Same architecture, same weights, plain PyTorch throughout (F.linear,
    F.scaled_dot_product_attention, a hand-written RMSNorm) -- the
    ground truth `llama_forward`'s kernel-built path is checked against.
    """
    B, N = input_ids.shape
    H = config.hidden_size
    cos, sin = precompute_rope(config, input_ids.device)

    def rmsnorm(h, weight):
        h_f32 = h.float()
        var = h_f32.pow(2).mean(-1, keepdim=True)
        return (h_f32 * torch.rsqrt(var + config.rms_eps)).to(h.dtype) * weight

    x = F.embedding(input_ids, weights.embed_tokens)
    residual = x

    for layer in weights.layers:
        normed = rmsnorm(residual, layer.input_layernorm_weight)

        q = F.linear(normed, layer.q_proj.t()).view(B, N, config.n_heads, config.head_dim).transpose(1, 2)
        k = F.linear(normed, layer.k_proj.t()).view(B, N, config.num_kv_heads, config.head_dim).transpose(1, 2)
        v = F.linear(normed, layer.v_proj.t()).view(B, N, config.num_kv_heads, config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, N)
        # repeat_kv, not SDPA's enable_gqa, so this stays a like-for-like
        # reference for llama_forward's kernel path (see module docstring).
        n_rep = config.n_heads // config.num_kv_heads
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, config.n_heads * config.head_dim)
        x = F.linear(attn_out, layer.o_proj.t())
        residual = residual + x

        normed2 = rmsnorm(residual, layer.post_attention_layernorm_weight)
        gate = F.linear(normed2, layer.gate_proj.t())
        up = F.linear(normed2, layer.up_proj.t())
        mlp_hidden = F.silu(gate) * up
        x = F.linear(mlp_hidden, layer.down_proj.t())
        residual = residual + x

    final_normed = rmsnorm(residual, weights.norm_weight)
    logits = F.linear(final_normed, weights.lm_head.t())
    return logits
