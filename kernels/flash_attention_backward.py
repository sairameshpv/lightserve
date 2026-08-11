"""Triton kernel 4b: FlashAttention backward pass (recomputation strategy).

Companion to flash_attention.py's forward-only kernel (kernel 4). The
backward pass never materializes the full [N, N] score/probability matrix
either -- same "O(N) memory, not O(N^2)" constraint as the forward -- by
*recomputing* each S_ij/P_ij tile from Q/K on the fly during the backward
sweep instead of ever having saved it from the forward pass. This is what
"recomputation strategy" means in the FlashAttention paper: forward saves
only O and one extra scalar per row (L, the log-sum-exp of that row's
scores) instead of the full attention matrix; backward uses that one scalar
to reconstruct the *exact* softmax probabilities it would have computed the
first time, not an approximation of them.

Algorithm (standard FlashAttention-2 backward, unchanged from the paper):

  1. Forward (this file's own kernel, not flash_attention.py's autotuned
     one -- see "why a separate forward kernel" below) additionally writes
     L_i = m_i + log(l_i) per query row alongside O. That's the one extra
     piece of state recomputation needs, O(N) memory (one float per row),
     not O(N^2).
  2. Preprocess kernel: D_i = rowsum(dO_i * O_i) per query row -- the
     softmax-Jacobian correction term. Cheap to precompute once instead of
     recomputing it inside every inner-loop step below.
  3. Main backward kernel, one program per K/V tile j (mirrors the
     forward's one-program-per-Q-tile shape, transposed): for each j, sweep
     every Q tile i, recompute S_ij = Q_i @ K_j^T * scale and
     P_ij = exp(S_ij - L_i) (exact -- L_i already encodes the full row's
     normalization, so this reproduces the same probabilities the forward
     pass's online-softmax loop produced, without ever having stored them),
     then:
       dV_j  += P_ij^T @ dO_i
       dP_ij  = dO_i @ V_j^T
       dS_ij  = P_ij * (dP_ij - D_i)
       dK_j  += dS_ij^T @ Q_i * scale
       dQ_i  += dS_ij @ K_j * scale
     dK_j/dV_j are exclusively owned by this program (it sweeps the
     *entire* Q range for its one j) so a plain store is enough at the end.
     dQ_i is touched by *every* j-program for a given (batch, head) --
     accumulated with tl.atomic_add into a float32 buffer (regardless of
     Q's own dtype -- same "tensor-core tl.dot, fp32 accumulate" idiom the
     forward kernel already uses) and cast down to Q's dtype only once, at
     the very end.

Why a separate forward kernel instead of adding an L output to
flash_attention.py's autotuned v2 kernel: that kernel is the one the
ours-vs-FA2 benchmark and ncu report (kernels/README.md) measure -- adding
an extra per-row store to it would perturb those already-recorded numbers
for a capability (backward) that benchmark was never scoped to cover. This
file's forward kernel is a plain, non-autotuned copy of the same
online-softmax loop (loop structure and math unchanged from v1/v2 -- see
flash_attention.py's docstring for the derivation) with one line added: the
final L_i store. `test_forward_matches_forward_only_kernel` (in the test
file) checks the two forward paths agree.

v1 scope, correctness not speed (same phasing as flash_attention.py's own
v1/v2 split): fixed (dtype/head-dim-dependent, see `_pick_block_sizes`)
tile sizes, no @triton.autotune, no causal early-exit on the non-causal
path (every K/V tile still visits every Q tile and masks per-element when
CAUSAL=False -- causal *does* skip Q tiles that are entirely above the
diagonal, since that's needed for the loop to terminate at the right place,
not just as a speed optimization). Gets the recomputation math right first;
an autotuned v2 is a natural follow-up, not done here.
"""
import torch
import triton
import triton.language as tl


def _pick_block_sizes(dtype: torch.dtype, D: int):
    """32x32 is flash_attention.py v1's proven-safe tile (fits the L40S's
    99KB/CTA cap at every dtype/head-dim combination tested there -- see
    kernels/README.md kernel 4 v1 section) -- but this file's backward
    kernel keeps more tiles resident per program at once (K, V, plus each
    step's Q, dO -- 4 big tiles) than the forward-only kernel (Q resident +
    K, V streamed -- effectively 3), so the same 32x32 that's safe there
    isn't always safe here. Confirmed on the L40S, not hand-estimated:
    fp32/D=128/32x32 raised `OutOfResources: Required 107008, Hardware
    limit 101376` running kernels/tests/test_flash_attention_backward.py --
    16x16 clears it with real margin. bf16 (half the bytes/element) keeps
    32x32 at every D this repo tests.
    """
    block_d = max(16, triton.next_power_of_2(D))
    if dtype.itemsize >= 4 and block_d > 64:  # fp32 (or wider), real D=128-class head dim
        return 16, 16
    return 32, 32


@triton.jit
def _flash_attn_fwd_lse_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    H, N, D,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Same online-softmax loop as flash_attention.py's forward kernel (see
    that file's docstring for the full derivation) plus one addition: the
    final m_i/l_i are combined into L_i = m_i + log(l_i) and written out.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_ptr += b * stride_qb + h * stride_qh
    k_ptr += b * stride_kb + h * stride_kh
    v_ptr += b * stride_vb + h * stride_vh
    o_ptr += b * stride_ob + h * stride_oh
    l_ptr += b * stride_lb + h * stride_lh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    hi = tl.minimum(N, (pid_m + 1) * BLOCK_M) if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        score_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            score_mask &= offs_m[:, None] >= offs_n[None, :]
        s = tl.where(score_mask, s, float("-inf"))

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

    acc = acc / l_i[:, None]
    # log(0) = -inf on fully-padded rows (l_i == 0 there) -- harmless, same
    # "padding rows never reach HBM unmasked" story as the forward-only
    # kernel's own acc/l_i divide: the mask below excludes those rows from
    # ever being stored.
    L_i = m_i + tl.log(l_i)

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)

    l_ptrs = l_ptr + offs_m * stride_lm
    tl.store(l_ptrs, L_i, mask=offs_m < N)


def _flash_attention_forward_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    """Forward pass that also returns L (per-row log-sum-exp, float32,
    shape [B, H, N]) -- the state the backward kernel below needs to
    recompute softmax probabilities without storing them. See module
    docstring for why this is a separate kernel from
    flash_attention.py's autotuned forward.
    """
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    scale = 1.0 / (D ** 0.5)
    BLOCK_D = max(16, triton.next_power_of_2(D))
    BLOCK_M, BLOCK_N = _pick_block_sizes(q.dtype, D)

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _flash_attn_fwd_lse_kernel[grid](
        q, k, v, o, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        H, N, D,
        scale,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o, L


@triton.jit
def _flash_attn_bwd_preprocess_kernel(
    o_ptr, do_ptr, d_ptr,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_db, stride_dh, stride_dm,
    H, N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """D_i = rowsum(dO_i * O_i) -- see module docstring point 2."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    o_ptr += b * stride_ob + h * stride_oh
    do_ptr += b * stride_dob + h * stride_doh
    d_ptr += b * stride_db + h * stride_dh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)

    o = tl.load(o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod, mask=mask, other=0.0)

    d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(d_ptr + offs_m * stride_dm, d, mask=offs_m < N)


@triton.jit
def _flash_attn_bwd_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    l_ptr, d_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    H, N, D,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program owns one K/V tile j and sweeps every Q tile i for its
    (batch, head) -- see module docstring point 3 for the per-step math.
    dK_j/dV_j are this program's alone (plain store at the end); dQ_i is
    shared across every j-program for this (batch, head), so it's
    accumulated with tl.atomic_add into a float32 buffer instead.
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_ptr += b * stride_qb + h * stride_qh
    k_ptr += b * stride_kb + h * stride_kh
    v_ptr += b * stride_vb + h * stride_vh
    do_ptr += b * stride_dob + h * stride_doh
    l_ptr += b * stride_lb + h * stride_lh
    d_ptr += b * stride_db + h * stride_dh
    dq_ptr += b * stride_dqb + h * stride_dqh
    dk_ptr += b * stride_dkb + h * stride_dkh
    dv_ptr += b * stride_dvb + h * stride_dvh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    kv_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)

    k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=kv_mask, other=0.0)
    v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=kv_mask, other=0.0)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Causal: K/V tile j only overlaps Q rows >= its own first column, so
    # start the Q sweep at that tile's floor instead of 0 -- the mirror of
    # the forward kernel's `hi` early-exit, from the other loop's
    # perspective (skip Q tiles entirely above the diagonal for every
    # column in this K/V tile; BLOCK_M == BLOCK_N here so this floor is
    # exact, not approximate). Non-causal still sweeps every Q tile --
    # correctness-first, no early-exit needed there, same v1 phasing as
    # flash_attention.py's forward kernel.
    lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M if CAUSAL else 0

    for start_m in range(lo, N, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)

        q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=q_mask, other=0.0)
        do = tl.load(do_ptr + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod, mask=q_mask, other=0.0)
        l_i = tl.load(l_ptr + offs_m * stride_lm, mask=offs_m < N, other=0.0)
        d_i = tl.load(d_ptr + offs_m * stride_dm, mask=offs_m < N, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # [BLOCK_M, BLOCK_N]

        score_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
        if CAUSAL:
            score_mask &= offs_m[:, None] >= offs_n[None, :]

        # Exact recomputation, not approximation: L_i is the *full* row's
        # log-sum-exp from the forward pass, so exp(s - L_i) reproduces the
        # same P_ij the forward pass's online-softmax loop produced for
        # this tile, without ever having stored it. Masked-out entries
        # (padding, or above the causal diagonal) go to 0 directly rather
        # than through exp(), so padding rows (l_i loaded as 0 via
        # other=0.0) can't turn into inf/NaN here.
        p = tl.where(score_mask, tl.exp(s - l_i[:, None]), 0.0)

        dv_acc += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")  # [BLOCK_M, BLOCK_N]
        ds = (p * (dp - d_i[:, None])).to(q.dtype)

        dk_acc += tl.dot(tl.trans(ds), q, input_precision="ieee") * scale

        dq_partial = tl.dot(ds, k, input_precision="ieee") * scale  # [BLOCK_M, BLOCK_D], fp32 (tl.dot accumulate)
        dq_ptrs = dq_ptr + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
        tl.atomic_add(dq_ptrs, dq_partial, mask=q_mask)

    dk_ptrs = dk_ptr + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_ptr + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk_acc.to(dk_ptr.dtype.element_ty), mask=kv_mask)
    tl.store(dv_ptrs, dv_acc.to(dv_ptr.dtype.element_ty), mask=kv_mask)


def _flash_attention_backward(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               o: torch.Tensor, L: torch.Tensor, causal: bool):
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    BLOCK_D = max(16, triton.next_power_of_2(D))
    BLOCK_M, BLOCK_N = _pick_block_sizes(q.dtype, D)

    # Row-sum(dO * O) precompute -- see module docstring point 2.
    Dsum = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    grid_pre = (triton.cdiv(N, BLOCK_M), B * H)
    _flash_attn_bwd_preprocess_kernel[grid_pre](
        o, do, Dsum,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        Dsum.stride(0), Dsum.stride(1), Dsum.stride(2),
        H, N, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )

    # float32 regardless of q's dtype: dQ is accumulated across multiple
    # K/V-tile programs via atomic_add (see kernel docstring), so it needs
    # a real global accumulator, not per-program registers -- fp32 keeps
    # that accumulation accurate the same way the forward kernel's tensor-
    # core tl.dot accumulates in fp32 regardless of bf16 inputs. Cast down
    # to q's dtype only once, after the kernel, to match what autograd
    # expects q.grad to look like.
    dq_fp32 = torch.zeros((B, H, N, D), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    grid_bwd = (triton.cdiv(N, BLOCK_N), B * H)
    _flash_attn_bwd_kernel[grid_bwd](
        q, k, v, do,
        L, Dsum,
        dq_fp32, dk, dv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        Dsum.stride(0), Dsum.stride(1), Dsum.stride(2),
        dq_fp32.stride(0), dq_fp32.stride(1), dq_fp32.stride(2), dq_fp32.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        H, N, D,
        scale,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return dq_fp32.to(q.dtype), dk, dv


class _FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        o, L = _flash_attention_forward_lse(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, L)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward(do, q, k, v, o, L, ctx.causal)
        return dq, dk, dv, None  # None: causal is a bool, not a tensor -- no gradient


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Autograd-enabled FlashAttention: same math as flash_attention.py's
    forward-only `flash_attention_forward`, plus a backward pass
    (recomputation strategy -- see module docstring). Use this entry point,
    not the forward-only one, whenever gradients are needed; use the
    forward-only one (as kernels/benchmark_flash_attention.py already does)
    when they aren't -- it's the autotuned, benchmarked-against-FA2 kernel,
    this one isn't.

    q, k, v: [B, H, N, D], same shape, same dtype, all on CUDA.
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, (
        f"expected q/k/v as [B, H, N, D], got {q.ndim}D/{k.ndim}D/{v.ndim}D"
    )
    assert q.shape == k.shape == v.shape, (
        f"q/k/v shapes must match, got {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}"
    )
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels need CUDA tensors"
    assert q.dtype == k.dtype == v.dtype, f"dtype mismatch: q={q.dtype} k={k.dtype} v={v.dtype}"

    return _FlashAttentionFunction.apply(q, k, v, causal)
