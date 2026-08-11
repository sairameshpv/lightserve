"""A minimal LLaMA-shaped decoder-only transformer, forward pass built from
this repo's own Triton kernels (kernels/tiled_matmul.py, kernels/
fused_rmsnorm_residual.py, kernels/flash_attention.py) instead of
torch.nn.Linear / F.rms_norm / F.scaled_dot_product_attention.

Scope, on purpose ("minimal"):

  - Plain multi-head attention, not LLaMA-3's real grouped-query attention
    (8 KV heads vs 32 Q heads) -- kernels/flash_attention.py's kernel
    requires q.shape == k.shape == v.shape (same H), so it can't express
    GQA's K/V-head-broadcast without a kernel change. hidden_size =
    n_heads * head_dim, every head count matches.
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
  - Random weights, not real LLaMA-3 checkpoint weights. This file is a
    systems/integration exercise (does the forward pass compose correctly,
    is it fast, does it graph-capture) not a quality one -- loading real
    HF weights would need a name-mapping layer with zero kernel-integration
    value added. `reference_llama_forward` (pure PyTorch, same random
    weights) is what correctness is checked against -- see
    model/tests/test_minimal_llama.py.

LLAMA3_8B_SHAPE below uses LLaMA-3-8B-Instruct's real per-layer dimensions
(hidden=4096, n_heads=32, head_dim=128, intermediate=14336, vocab=128256 --
the same model benchmarks/run_baseline.py and kernels/benchmark_flash_
attention.py already target) with `n_layers` overridable -- the real model
has 32; running all 32 isn't necessary to demonstrate the integration or
the CUDA-graph mechanics in model/cuda_graph_decode.py, so that script
defaults to a handful of layers and lets you scale up.
"""
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
    rope_theta: float = 500000.0  # LLaMA-3's real value (LLaMA-1/2 used 10000)
    rms_eps: float = 1e-5
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        assert self.hidden_size == self.n_heads * self.head_dim, (
            f"hidden_size ({self.hidden_size}) must equal n_heads*head_dim "
            f"({self.n_heads}*{self.head_dim}={self.n_heads * self.head_dim}) -- "
            "plain MHA only here, see module docstring on GQA"
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

    layers = []
    for _ in range(config.n_layers):
        layers.append(LayerWeights(
            input_layernorm_weight=ones(H),
            q_proj=randn(H, qkv_dim),
            k_proj=randn(H, qkv_dim),
            v_proj=randn(H, qkv_dim),
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


def precompute_rope(config: LlamaConfig, device: str):
    """cos/sin lookup tables, [max_seq_len, head_dim] each -- a fixed buffer
    sliced identically on every call (same [0:N] range every decode step in
    this file's "always run the full max_seq_len buffer" design, see
    cuda_graph_decode.py), so this needs no dynamic control flow and is
    trivially CUDA-graph-safe.
    """
    half = config.head_dim // 2
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
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
        k = _linear(normed_2d, layer.k_proj).reshape(B, N, config.n_heads, config.head_dim).transpose(1, 2)
        v = _linear(normed_2d, layer.v_proj).reshape(B, N, config.n_heads, config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, N)

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
        k = F.linear(normed, layer.k_proj.t()).view(B, N, config.n_heads, config.head_dim).transpose(1, 2)
        v = F.linear(normed, layer.v_proj.t()).view(B, N, config.n_heads, config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin, N)

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
