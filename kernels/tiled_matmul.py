"""Triton kernel 3: tiled GEMM (C = A @ B) with shared-memory-resident tiles.

Kernels 1 and 2 were memory-traffic stories (elementwise / row-reduction,
FLOPs were negligible next to HBM round-trips). GEMM is the opposite: for
[M,K]@[K,N], FLOPs grow as O(M*K*N) but bytes only as O(M*K + K*N + M*N), so
past a modest size a GEMM is compute-bound (see the roofline analysis in
../benchmarks/profiling/roofline/ -- FFN/QKVO cross into compute-bound
territory once batched past a few hundred tokens). The whole game here is
reusing each loaded element for many multiply-adds before it leaves the
chip, which is exactly what tiling buys you:

  Naive matmul: for each output C[i,j], stream the whole i-th row of A and
  j-th column of B from HBM. Every element of A gets re-read N times (once
  per output column); every element of B gets re-read M times.

  Tiled matmul: split A/B/C into BLOCK_M x BLOCK_K, BLOCK_K x BLOCK_N,
  BLOCK_M x BLOCK_N tiles. Each program loads ONE tile of A and ONE tile of B
  per K-step into on-chip memory, then does BLOCK_M*BLOCK_N*BLOCK_K
  multiply-adds against those same on-chip tiles before moving on -- each
  loaded element gets reused BLOCK_N (for A) or BLOCK_M (for B) times before
  it's evicted, instead of being re-read from HBM every time.

If you're coming from raw CUDA: a hand-written tiled GEMM stages each tile
through `__shared__` arrays explicitly, with manual `__syncthreads()` between
the load and compute phases. Triton doesn't expose shared memory as a
distinct address space you allocate -- `tl.load()`-ing a tile that's about to
feed `tl.dot()` is where the compiler places it on-chip (visible if you
inspect the generated PTX/SASS, or the smem-related counters in
`ncu`), and it's also where it schedules the load-vs-compute pipelining
(`num_stages` below) that raw CUDA does by hand with double-buffered shared
memory. `tl.dot` itself compiles to real Tensor Core MMA instructions on
Ada (this GPU) when the operand dtypes qualify (bf16/fp16 in, fp32
accumulate) -- the same hardware path cuBLAS uses. See
`matmul_ncu_report.md` for what that gap with cuBLAS actually is and isn't
about (spoiler: it's not "cuBLAS uses tensor cores and this doesn't").
"""
import torch
import triton
import triton.language as tl


# A handful of tile-size/pipelining configs, not an exhaustive search --
# cuBLAS's real advantage is picking from (and having hand-tuned in SASS) far
# more configs than this across far more shapes; this is enough to be a
# genuinely tiled, genuinely tensor-core GEMM, not enough to match cuBLAS's
# engineering effort. See the writeup for what the remaining gap is made of.
_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=5, num_warps=4),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Grid "swizzling" for L2 locality: the obvious row-major mapping
    # (pid -> (pid // grid_n, pid % grid_n)) processes tiles in an order
    # where consecutive programs share almost nothing -- program 0 and
    # program 1 use the same A-row-tile but completely different B-column-
    # tiles, so B tiles get evicted from L2 before they'd ever be reused.
    # Grouping GROUP_M row-tiles together and iterating N *within* the group
    # means consecutive programs reuse the same handful of A tiles, so the
    # ones still resident in L2 actually get hit again -- pure scheduling,
    # no extra compute, and it's worth several tens of percent of achieved
    # TFLOPS on its own. (This is the "L2-cache reuse" half of what a tiled
    # GEMM needs beyond just on-chip tiling for HBM traffic.)
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # fp32 accumulator regardless of input dtype -- same reasoning as kernel
    # 2's variance: bf16/fp16 partial sums over K steps up to 8192+ elements
    # lose real precision otherwise. This is also what cuBLAS/Tensor Cores
    # do natively (MMA instructions accumulate in fp32 even for bf16/fp16
    # operands), not an extra cost we're paying that cuBLAS skips.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # Both the K edge (last, possibly-partial K-tile) AND the M/N edge
        # (last, possibly-partial row/col tile) need masking here -- a mask
        # of shape [1, BLOCK_K] broadcasts across every row regardless of
        # whether that row is in-bounds, so leaving out the offs_m/offs_n
        # term reads past the tensor's actual allocation for out-of-range
        # rows/cols on every K-step, not just once. The masked final store
        # below means that garbage never reaches C, but reading unowned
        # memory on every iteration is still real undefined behavior worth
        # not doing, not just a wasted load.
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        # tl.dot -> Tensor Core MMA on Ada when a/b are bf16/fp16. This is
        # the payoff for tiling: BLOCK_M*BLOCK_N*BLOCK_K FMAs against tiles
        # that are already on-chip, not re-fetched from HBM per FMA.
        # input_precision="ieee" only changes anything for fp32 operands
        # (bf16/fp16 have one native MMA precision, this is a no-op for
        # them) -- tl.dot's default for fp32 is actually TF32 (truncated
        # mantissa, real tensor-core throughput), which does NOT match
        # torch.matmul's fp32 default (strict IEEE on this PyTorch/driver
        # combination -- allow_tf32 defaults to False). Without this, the
        # fp32 correctness tests fail against cuBLAS by up to ~0.05 absolute
        # on a K=256+ reduction: not a bug in either implementation, just two
        # different (both legitimate) precision modes being compared as if
        # they were the same one. This kernel's bf16 benchmark path is
        # unaffected either way -- bf16 has no such ambiguity.
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B. a: [M, K], b: [K, N], same dtype, both on CUDA."""
    assert a.ndim == 2 and b.ndim == 2, f"expected 2D matrices, got {a.ndim}D and {b.ndim}D"
    assert a.shape[1] == b.shape[0], f"inner dims must match: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
    assert a.is_cuda and b.is_cuda, "Triton kernels need CUDA tensors"
    assert a.dtype == b.dtype, f"dtype mismatch: a={a.dtype} b={b.dtype}"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c