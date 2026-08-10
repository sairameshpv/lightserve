"""Triton kernel 4: FlashAttention forward pass (v1 — correctness, not speed).

Computes O = softmax(Q @ K^T * scale) @ V for q, k, v: [B, H, N, D], without
ever materializing the [N, N] score matrix — see the plan this implements
(tile sizes / SRAM budget / loop structure) for the derivation. This file is
deliberately v1: fixed block sizes (no `@triton.autotune`), fp32 in/out, no
backward pass. The whole point of v1 is getting the online-softmax loop
structure correct and allclose-verified against PyTorch's SDPA before
touching anything performance-related (tensor-core dtypes, causal
early-exit, autotuned tile sizes) in a v2.

Loop structure, one Triton program per query tile:

    load Q_i                                    # [BLOCK_M, D], resident whole loop
    m_i, l_i, acc = -inf, 0, 0                   # running max / softmax denom / output accumulator
    for each K/V tile (BLOCK_N wide), sequentially:
        S_ij  = Q_i @ K_j^T * scale              # tl.dot -> [BLOCK_M, BLOCK_N]
        m_new = max(m_i, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)                # unnormalized probs, this tile only
        alpha = exp(m_i - m_new)                 # rescale factor for the *old* accumulator
        l_i   = alpha * l_i + rowsum(P_ij)
        acc   = alpha * acc + P_ij @ V_j          # second tl.dot
        m_i   = m_new
    O_i = acc / l_i                              # single normalization at the end

This is the "online softmax" trick: because softmax needs the max/sum over
the *whole* row before it can normalize anything, and that whole row would
be an [N] value we don't have yet after only one K/V tile, each step folds
in a new tile's contribution and *rescales* the running accumulator by how
much the running max moved -- so the accumulator is always consistent with
"softmax so far," and the final division is the only place normalization
actually happens.
"""
import torch
import triton
import triton.language as tl


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
    # slice of one (batch, head)'s queries, and sweeps the *entire* K/V
    # sequence for that (batch, head) sequentially in the loop below. Unlike
    # kernel 3's grid (which tiles both output axes so every program is
    # independent), FlashAttention's inner loop is inherently sequential --
    # each step's rescale depends on the previous step's m_i/l_i/acc.
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

    # v1: loop over the FULL key sequence every time, even for causal, and
    # let the score mask below zero out the upper-triangular tiles. A causal
    # early-exit (stop once start_n > this tile's query range, per the plan)
    # is a real ~2x FLOPs win but it's a speed change, not a correctness
    # one -- deliberately deferred to v2 so this loop has exactly one job
    # (get online softmax right) at a time.
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # input_precision="ieee": same reasoning as kernel 3 -- tl.dot's
        # default for fp32 operands is TF32 (truncated mantissa), which
        # would not match torch's SDPA reference at the tolerances this v1
        # is verified against. bf16/fp16 (v2) have no such ambiguity.
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # [BLOCK_M, BLOCK_N]

        score_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            score_mask &= offs_m[:, None] >= offs_n[None, :]
        s = tl.where(score_mask, s, float("-inf"))

        # Online softmax update. For a fully-out-of-bounds query row (only
        # possible when N isn't a multiple of BLOCK_M, i.e. offs_m >= N)
        # every s in that row is -inf on every tile, so m_i/m_new both stay
        # -inf and alpha/p below evaluate to NaN for that row specifically.
        # That's fine, not UB: no memory is touched out of bounds (loads are
        # masked with other=0.0), the NaN is pure register arithmetic, and
        # the final store below masks that row out before it ever reaches
        # HBM -- same "compute garbage, never store it" pattern kernel 3
        # uses for its edge tiles.
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

    # No divide-by-zero guard needed here: every *valid* query row (offs_m <
    # N) has l_i > 0 -- causal always keeps the diagonal key (offs_n ==
    # offs_m) unmasked, non-causal keeps every in-bounds key unmasked, so
    # rowsum(p) is never all-zero for a real row. Padding rows (offs_m >= N)
    # get NaN here (-inf - -inf = NaN propagates from m_i through alpha and
    # p, same as the loop comment above), not a clean 0/0 -- but they're
    # masked out at the store below, so it never reaches HBM.
    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 32,
    block_n: int = 32,
) -> torch.Tensor:
    """O = softmax(Q @ K^T / sqrt(D)) @ V, tiled with online-softmax accumulation.

    q, k, v: [B, H, N, D], same shape, same dtype, all on CUDA. v1: fixed
    block sizes, no autotune, correctness-only -- see module docstring.

    Defaults (block_m=block_n=32) are deliberately conservative, not tuned --
    picked to fit the L40S's 99KB/CTA shared-memory cap (`OutOfResources`
    from Triton itself otherwise) up through D=128 real head dims, per the
    SRAM-budget plan this kernel implements. 64x64 at D=128 measured at
    ~176KB required vs 99KB available; the plan's hand-estimate (96KB at
    Br=Bc=64) was optimistic because it didn't account for Triton's own
    pipelining/staging overhead on top of the Q/K/V/O/S working set. Bigger
    tiles are a real speed lever for v2's autotune sweep, bounded by this
    same cap.
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
    # dim on-chip, same whole-row-at-once approach as kernel 2's BLOCK_N.
    # Clamped to >=16: tl.dot requires its contraction dim >= 16 (Q@K^T
    # contracts over D), so a small real head dim (e.g. D=8) would otherwise
    # violate that and fail to compile, not just run inefficiently.
    BLOCK_D = max(16, triton.next_power_of_2(D))

    grid = (triton.cdiv(N, block_m), B * H)
    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N, D,
        scale,
        CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
    )
    return o
