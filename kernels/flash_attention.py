"""Triton kernel 4: FlashAttention forward pass.

v1 (see git history / kernels/README.md) was correctness-only: fixed 32x32
tile, fp32 in/out, no causal early-exit -- the point was getting the
online-softmax loop structure right before touching anything
performance-related. This is v2, the performance pass, on top of the exact
same loop structure:

  - `@triton.autotune` over BLOCK_M, BLOCK_N, num_warps, num_stages (a
    handful of configs, not an exhaustive search -- same "genuinely tuned,
    not cuBLAS-level engineering effort" spirit as kernel 3's autotune).
  - Causal early-exit: the K/V loop now stops once past a query tile's
    diagonal (`hi = min(N, (pid_m+1)*BLOCK_M)`) instead of sweeping the
    full sequence and masking the upper triangle away after the fact --
    roughly halves FLOPs for causal (prefill-shaped) attention, matching
    what FlashAttention-2 itself does and what a fair TFLOPS-vs-FA2
    comparison requires.
  - bf16 is the dtype this is actually tuned for (tensor-core `tl.dot`,
    fp32 accumulate) -- v1's fp32 path still works (same kernel, dtype-
    generic) but was SRAM-constrained: fp32 Q/K/V tiles are 2x the bytes
    of bf16 ones, which is *why* v1 was capped at 32x32. bf16 tiles free up
    that same 99KB/CTA budget for the bigger tiles this file tunes over.

Loop structure and online-softmax math are UNCHANGED from v1 -- see that
docstring (in git history) or kernels/README.md kernel 4 section for the
derivation. This file's docstring only covers what's new.
"""
import torch
import triton
import triton.language as tl


# A handful of tile-size/pipelining configs, not an exhaustive search -- see
# kernel 3's _CONFIGS comment for the same reasoning. BLOCK_M/BLOCK_N/
# num_stages/num_warps span the usual tile-size-vs-occupancy-vs-pipelining
# tradeoff kernel 3's ncu report already explored for a plain GEMM.
# {"BLOCK_M": 32, "BLOCK_N": 32} (stages=2) is the one deliberately-small
# entry: it's the fp32 safety net (see _prune_by_shared_mem below) --
# without it, no config here fits an fp32 call at a real D=128 head dim at
# all, and autotune would crash outright instead of falling back to
# something merely slow.
_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=4),
]


def _prune_by_shared_mem(configs, named_args, **kwargs):
    """Filters _CONFIGS down to ones that actually fit the L40S's 99KB/CTA
    shared-memory cap for this call's dtype and head dim.

    This has to exist because triton.autotune does NOT skip a config that
    fails to compile -- it raises OutOfResources and crashes the whole
    search the first time it tries one, same failure v1 hit at a single
    fixed tile size (see kernels/README.md). Every config here was tuned
    for bf16 at real head dims; fp32 tiles are 2x the bytes (same v1
    lesson, one more time at v2's larger tile sizes), so most of them don't
    fit fp32 *at all* once BLOCK_D (next_power_of_2(D), clamped >=16)
    exceeds 64 -- confirmed by actually compiling every (config, dtype,
    BLOCK_D) combination on the real GPU rather than hand-estimating (the
    v1 estimate undercounted Triton's own pipelining overhead once already;
    no reason to trust a second hand-estimate at bigger tiles). Full
    probe table is in kernels/README.md's v2 section.
    """
    itemsize = named_args["q_ptr"].element_size()  # 4 = fp32, 2 = bf16/fp16
    block_d = max(16, triton.next_power_of_2(named_args["D"]))
    if itemsize >= 4:  # fp32: only the dedicated small safety-net config fits at all once BLOCK_D>64
        safe = {(32, 32)}
    elif block_d <= 64:  # bf16, small head dim: everything but 64x128 fits
        safe = {(128, 64), (128, 128), (64, 64), (64, 32), (32, 64), (32, 32)}
    else:  # bf16, real D=128-class head dim
        safe = {(128, 64), (64, 32), (32, 32)}
    kept = [c for c in configs if (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"]) in safe]
    return kept or configs[-1:]  # never return empty; fall back to the smallest config over crashing


@triton.autotune(configs=_CONFIGS, key=["N", "D"], prune_configs_by={"early_config_prune": _prune_by_shared_mem})
@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, N, D,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Grid is (query_tile, batch*head) -- one program owns one BLOCK_M-tall
    # slice of one (batch, head)'s queries, and sweeps K/V for that
    # (batch, head) sequentially in the loop below. Unlike kernel 3's grid
    # (every program independent), FlashAttention's inner loop is inherently
    # sequential -- each step's rescale depends on the previous step's
    # m_i/l_i/acc.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_ptr += b * stride_qb + h * stride_qh
    k_ptr += b * stride_kb + h * stride_kh
    v_ptr += b * stride_vb + h * stride_vh
    o_ptr += b * stride_ob + h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Q tile is loaded once and stays resident (in registers) for the whole
    # K/V loop below -- this is the "Q_i" from the plan, reused BLOCK_N-many
    # times per tile instead of being re-streamed from HBM.
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Causal early-exit: any key tile entirely past this query tile's
    # diagonal is 100% masked anyway (every offs_m in this program is <
    # hi <= every offs_n a later tile would have), so skip it instead of
    # computing two tl.dots' worth of work just to multiply the result by
    # zero. The boundary tile (the one straddling the diagonal) is still
    # visited -- it still needs the per-element mask below, unchanged from
    # v1. This is the only behavioral difference from v1's loop; softmax
    # correctness doesn't depend on it (v1 got the same answer, slower).
    hi = tl.minimum(N, (pid_m + 1) * BLOCK_M) if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # input_precision="ieee" only matters for fp32 operands (forces
        # strict IEEE instead of tl.dot's TF32-truncated default, needed to
        # match an fp32 SDPA reference -- see kernel 3). It's a no-op for
        # bf16/fp16, which is the dtype this file is actually tuned for --
        # tl.dot compiles to real Tensor Core MMA there, fp32-accumulated
        # either way.
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # [BLOCK_M, BLOCK_N]

        score_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            score_mask &= offs_m[:, None] >= offs_n[None, :]
        s = tl.where(score_mask, s, float("-inf"))

        # Online softmax update -- see v1 docstring (git history) for the
        # full derivation and the NaN-on-padding-rows note, unchanged here.
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = alpha * l_i + tl.sum(p, axis=1)

        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    # No divide-by-zero guard needed: every valid query row has l_i > 0
    # (causal always keeps the diagonal key unmasked, non-causal keeps
    # every in-bounds key unmasked). Padding rows get NaN, masked out below
    # before it reaches HBM -- see v1 docstring for why.
    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


def flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """O = softmax(Q @ K^T / sqrt(D)) @ V, tiled with online-softmax accumulation.

    q, k, v: [B, H, N, D], same shape, same dtype, all on CUDA. Tile sizes,
    num_warps, and num_stages are autotuned (see _CONFIGS) -- bf16 is the
    dtype this is tuned for (Tensor Core tl.dot); fp32 still works
    (correctness path, same as v1) but wasn't part of the tuning sweep and
    will be slower relative to its own peak than the bf16 numbers in
    kernels/README.md.
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, (
        f"expected q/k/v as [B, H, N, D], got {q.ndim}D/{k.ndim}D/{v.ndim}D"
    )
    assert q.shape == k.shape == v.shape, (
        f"q/k/v shapes must match, got {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}"
    )
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels need CUDA tensors"
    assert q.dtype == k.dtype == v.dtype, f"dtype mismatch: q={q.dtype} k={k.dtype} v={v.dtype}"

    B, H, N, D = q.shape
    o = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)
    # D isn't tiled (see the plan) -- one program loads/keeps the whole head
    # dim on-chip. Clamped to >=16: tl.dot requires its contraction dim >=
    # 16 (Q@K^T contracts over D), so a small real head dim would otherwise
    # violate that and fail to compile, not just run inefficiently.
    BLOCK_D = max(16, triton.next_power_of_2(D))

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N, D,
        scale,
        CAUSAL=causal,
        BLOCK_D=BLOCK_D,
    )
    return o
