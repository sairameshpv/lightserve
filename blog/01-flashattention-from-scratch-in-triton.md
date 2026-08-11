# FlashAttention from scratch in Triton

*Part 1 of a series on building an inference stack's kernels from the ground up. Code: [lightserve](https://github.com/sairameshpv/lightserve), specifically `kernels/flash_attention.py`, `kernels/flash_attention_backward.py`, and `model/`. Everything below ran on a real L40S, not simulated or hand-estimated.*

Attention is the one op in a transformer you can't just hand off to cuBLAS. `softmax(QKᵀ/√d)V` looks like two matmuls with a softmax sandwiched in the middle, and if you write it that way — materialize the full `[N, N]` score matrix, softmax it, multiply by `V` — you'll blow your memory budget long before you blow your compute budget. At `N=8192`, `[N,N]` in fp32 is 256MB *per head*. FlashAttention's whole contribution is refusing to ever materialize that matrix, and doing the softmax in a single streaming pass instead. This post is about building that kernel — forward, backward, and then actually wiring it into a model — in Triton, on real hardware, with the real bugs that showed up along the way.

## The trick: online softmax

Softmax needs a full row before it can normalize anything — `exp(x_i) / Σexp(x_j)` needs every `x_j` in the row before you can safely compute even one output. That's what forces the naive implementation to materialize the whole row (and hence the whole `[N,N]` matrix) before it can do anything. The way around it is to keep a *running* estimate of the row's max and sum, and correct your previous partial answer every time a new tile shifts that estimate:

```
m_new = max(m_i, max(s_tile))          # running max, updated
alpha = exp(m_i - m_new)               # how much the max moved
l_i   = alpha * l_i + sum(exp(s_tile - m_new))   # running sum, rescaled
acc   = acc * alpha + exp(s_tile - m_new) @ v_tile  # running output, rescaled
```

Every time a new K/V tile comes in, you correct the accumulator for what you would have computed differently if you'd known the new max from the start, then fold the new tile in. By the time you've swept every K/V tile, `acc / l_i` is the exact softmax-weighted output — not an approximation, just computed incrementally. This is the whole idea; everything else is tiling mechanics on top of it.

## v1: get the loop right before anything else

The first version of `kernels/flash_attention.py` had one job: prove the online-softmax loop above actually produces the right answer, before touching anything about speed. Fixed 32×32 tiles, fp32 only, no autotuning, no causal short-circuiting (it masked the upper triangle but still swept every K/V tile even for causal attention). Boring on purpose.

It still found two real bugs the moment it hit the GPU:

**`tl.dot` needs a contraction dimension ≥16.** A `D=8` test shape (deliberately picked to exercise the "degenerate small head-dim" edge) gave `BLOCK_D = next_power_of_2(8) = 8`, which is the K-dimension of `Q @ Kᵀ` — Triton doesn't allow that: `AssertionError: Input shapes should have M >= 1, N >= 1 and K >= 16`. Fixed by clamping `BLOCK_D = max(16, next_power_of_2(D))`.

**64×64 tiles at a real head dim blow the shared-memory budget.** `BLOCK_M=BLOCK_N=64` at `D=128` needed 180,480 bytes of shared memory against the L40S's 101,376-byte-per-CTA cap — `OutOfResources`. A pre-implementation estimate had put `Br=Bc=64` comfortably inside the budget by hand; the estimate was wrong, mostly because it didn't account for Triton's own load/compute pipelining overhead on top of the raw Q/K/V/O/S working set. Fixed by dropping to 32×32, which fits everywhere tested. Lesson taken forward into every kernel after this one: verify shared-memory budgets by actually compiling on the real GPU, don't trust the arithmetic.

15 tests passed at that point (27 once v2 added bf16), correctness-only, no speed claims.

## v2: making it fast

Three changes on top of the same, unchanged loop:

- **`@triton.autotune`** over `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` — 7 hand-picked configs, not an exhaustive search.
- **Causal early-exit**: the K/V loop now stops once it's past a query tile's diagonal (`hi = min(N, (pid_m+1)*BLOCK_M)`) instead of sweeping the whole sequence and masking the upper triangle away afterward. Roughly halves the FLOPs for causal attention — which is every prefill call in a real decoder-only model.
- **bf16**, tensor-core `tl.dot`, fp32 accumulate.

This surfaced a subtler bug: `triton.autotune` doesn't skip a config that fails to compile, it crashes the *entire* search the first time it hits one. Since the 7 configs were picked for bf16, most of them don't fit an fp32 call at all once the head dim gets real (`D=128`) — fp32 tiles are twice the bytes of bf16. Rather than hand-estimate a second time (the v1 estimate was already wrong once), every `(config, dtype, head_dim)` combination got actually compiled on the L40S to find out which configs survive, baked into a `prune_configs_by` hook. At `D=128`/bf16, only 3 of the 7 configs fit at all; at `D=128`/fp32, only one dedicated small safety-net config does.

With that in place, here's ours against PyTorch's real FlashAttention-2 (forced onto the `FLASH_ATTENTION` SDPA backend — not an approximation of FA2, the actual kernel), bf16, LLaMA-3-8B's real head shape (`H=32, D=128`):

| shape (B,H,N,D) | causal | ours TFLOPS | FA2 TFLOPS | % of FA2 | % of peak |
|---|---|--:|--:|--:|--:|
| 1×32×512×128 | causal | 40.9 | 47.7 | 85.7% | 11.3% |
| 1×32×1024×128 | causal | 102.0 | 137.6 | 74.1% | 28.2% |
| 1×32×2048×128 | causal | 125.6 | 173.1 | 72.6% | 34.7% |
| 1×32×4096×128 | causal | 146.2 | 146.8 | **99.6%** | 40.4% |
| 1×32×8192×128 | causal | 166.6 | 155.7 | **107.0%** | 46.0% |

Every shape clears 60% of FA2's throughput, and by 4096–8192 tokens we're at parity or ahead. The Nsight Compute profile explains why the gap closes rather than stays fixed: occupancy is a wash (both kernels land at ~8 resident warps/SM despite different block-size/thread-count tradeoffs), and zero shared-memory bank conflicts on the load side for ours — the real difference is **Tensor Core pipe utilization** (33.4% vs 45.9% at N=1024, 57.6% vs 76.9% at N=8192). FA2's hand-written CUTLASS inner loop schedules tensor-core instructions more efficiently per cycle than a 7-config Triton sweep reaches — the same kind of engineering-effort gap kernel 3 (a plain tiled GEMM) found against cuBLAS. More total work amortizes per-launch overhead, which is why the gap shrinks as N grows rather than staying constant.

## The backward pass: recomputation, not storage

Training needs gradients, and the memory constraint that motivated the forward pass doesn't go away for the backward pass — you still can't materialize `[N,N]`. FlashAttention's answer here is called the "recomputation strategy," and it's a clean idea once you see it: the forward pass saves only the output `O` and *one extra scalar per row* — `L = m + log(l)`, that row's log-sum-exp — instead of the softmax probability matrix. The backward pass then recomputes each score tile from `Q`/`K` on the fly, and recovers the *exact* same softmax probabilities via `P = exp(S - L)`, because `L` already encodes everything the normalization needed to know. Nothing approximate about it — it's the same numbers, just not the ones sitting in memory.

The per-step math (standard FlashAttention-2 backward, one program per K/V tile, sweeping every Q tile):

```
S_ij  = Q_i @ K_jᵀ * scale
P_ij  = exp(S_ij - L_i)                    # exact recompute, using the saved L
dV_j += P_ijᵀ @ dO_i
dP_ij = dO_i @ V_jᵀ
dS_ij = P_ij * (dP_ij - D_i)               # D_i = rowsum(dO_i * O_i), precomputed once
dK_j += dS_ijᵀ @ Q_i * scale
dQ_i += dS_ij @ K_j * scale
```

`dK_j`/`dV_j` are owned entirely by the program handling K/V tile `j` — it sweeps the whole Q range itself, so a plain store at the end is enough. `dQ_i` is different: every K/V-tile program touches it (every `j` contributes to every `i`'s gradient), so it has to be accumulated across programs with `tl.atomic_add`, into a float32 buffer regardless of Q's own dtype — the same "tensor-core dot, fp32 accumulate" idiom the forward kernel already leans on, just applied to a cross-program accumulator instead of a per-program register.

This is where the second real bug showed up. The backward kernel keeps *more* tiles resident per program than the forward kernel — K, V, and each step's Q and dO (4 big tiles), against the forward's Q-resident-plus-streamed-K/V (effectively 3). The exact 32×32 tile size that v1's shared-memory probe proved safe for fp32 at `D=128` wasn't safe here: `OutOfResources: Required 107008, Hardware limit 101376`. Close, but over. Fixed with a dtype-and-head-dim-aware tile size — 16×16 for fp32 at `D>64`, 32×32 otherwise — the same "verify on real hardware" habit as v1, applied to a kernel with a different resource footprint.

Correctness here got checked two independent ways: `torch.autograd.gradcheck` (numerical finite-difference vs. the analytical gradient — the standard way to validate a custom `backward()` is the true Jacobian of its `forward()`, and it doesn't trust any reference attention implementation, only calculus), and dQ/dK/dV compared directly against PyTorch's own autograd through SDPA. Both passed, 23/23 tests, all 4 `gradcheck` cases included.

Speed is a different story, and an honest one to tell. This backward kernel is still "v1" — fixed tiles, no autotuning — benchmarked against FA2's tuned backward:

| shape (B,H,N,D) | causal | ours TFLOPS | FA2 TFLOPS | % of FA2 |
|---|---|--:|--:|--:|
| 1×32×512×128 | causal | 11.4 | 44.5 | 25.5% |
| 1×32×1024×128 | causal | 13.9 | 100.2 | 13.8% |
| 1×32×2048×128 | causal | 15.0 | 122.4 | 12.2% |
| 1×32×4096×128 | causal | 15.6 | 138.1 | 11.3% |

Ours plateaus around 14–16 TFLOPS no matter what N does; FA2 keeps climbing to 138. That's exactly what a fixed-tile kernel with no autotuning search looks like next to one that has both — there's no lever to pull as the problem grows, so throughput stays flat while FA2's larger effective tiles keep extracting more. This is the same gap the *forward* kernel's own v1 would have shown against FA2, had anyone benchmarked it (nobody did — v1 was scoped to correctness only, on purpose). An autotuned backward v2 is the obvious next step and isn't done here.

## Putting it in a model, and the bug that only shows up there

Individually-correct kernels don't guarantee a correct model — composition has its own failure modes, and building `model/minimal_llama.py` (a LLaMA-shaped forward pass built entirely from these kernels: `matmul` for every linear, `fused_add_rmsnorm` for every pre-norm, `flash_attention_forward` for every attention call) found one immediately.

Pre-norm transformers fold each sublayer's output into the residual stream at the start of the *next* sublayer's norm call — `fused_add_rmsnorm(x, residual, weight)` computes `h = x + residual` and normalizes `h`, so the "add" naturally happens as a side effect of the next norm. That's an elegant way to avoid a separate add op — until you hit the *last* layer, which has no next norm call to do that fold for it. The code originally passed `torch.zeros_like(x)` into the final norm instead of the real last-layer MLP output, on the theory that there was "nothing left to add." There was: the last layer's own MLP output, silently discarded. On a 1-layer test model this meant 96.8% of output elements disagreed with a plain-PyTorch reference — not a subtle numerical drift, a component of the forward pass that never ran. The fix is one line: feed the real `x`, not zeros, into the final norm call. The bug is worth dwelling on because it's the kind that only exists at the composition level — every kernel involved was individually correct.

## CUDA graphs on the decode loop

The last piece: wrapping one autoregressive decode step in a `torch.cuda.CUDAGraph` and measuring what that actually buys. The premise is that batch-1 decode does very little GPU work per kernel — one token's worth — so the CPU time spent *issuing* each of the dozens of kernel launches per step can rival the GPU time spent running them. A captured graph replaces that whole launch sequence with a single `graph.replay()` call.

Getting a real incremental KV cache into this would need the FlashAttention kernel above to accept a query length different from its key/value length (`Q` = 1 new token, `K`/`V` = a growing cache) — which it doesn't; `flash_attention_forward` asserts `q.shape == k.shape == v.shape`. Rather than extend the kernel, the decode loop here keeps a fixed-size `[B, max_seq_len]` buffer and reruns the *full* causal self-attention over it every step. That sounds wasteful, and it is — real production stacks solve the shape-staticness problem with incremental caches, not full recompute — but it composes with the existing kernel unmodified, and it's legitimately correct: causal masking guarantees a row can never read a column past itself, so the not-yet-real positions further in the buffer never contaminate the one row actually read each step.

At LLaMA-3-8B's real per-layer shape (`hidden=4096, heads=32, head_dim=128, intermediate=14336`), 4 layers, batch=1:

| n_layers | kernels/forward call | eager median | cuda_graph median | speedup |
|---|--:|--:|--:|--:|
| 4 | 108 | 5.966 ms | 5.219 ms | 1.14× |
| 1 | 42 | 3.196 ms | 2.507 ms | 1.28× |

Modest, and smaller at more layers rather than larger — worth explaining rather than just reporting. This design reruns real `4096×14336` matmuls at `N=128` every step, so a genuine chunk of the measured time is GEMM compute, not launch overhead, and that compute grows with layer count while the CPU overhead CUDA graphs remove stays roughly proportional to kernel *count* — more layers dilutes the relative benefit. (The first attempt at even *measuring* kernel count hit its own bug: profiling a cold, un-warmed-up forward call reported 7,765 kernel launches, which was really counting every matmul's one-time `@triton.autotune` search as if every decode step paid for it. Warming up before profiling brought that down to a sane 108.) A real incremental-KV-cache decode step would do genuinely single-token-sized compute, making launch overhead a *larger* fraction of a much smaller total — that's where CUDA graphs are expected to earn their keep, and it's flagged as the natural next step rather than claimed here.

## What actually held up, and what's next

Every real bug in this post — the `tl.dot` minimum contraction dimension, two separate shared-memory overflows (forward and backward, different tile-occupancy reasons), the autotune-crashes-on-first-failure behavior, and the dropped final-layer residual — showed up by actually running on the L40S, not by reasoning about the code. None of them were visible from the math. That's the recurring theme worth taking away more than any single number: verify on real hardware, and when a benchmark shows a gap, explain the gap instead of hiding it.

What's flagged, not done: an autotuned backward v2 (this post's backward benchmark is exactly the gap you'd expect from a fixed-tile kernel against a tuned one); grouped-query attention, which real LLaMA-3-8B uses and this integration doesn't (the FA kernel would need to broadcast K/V across a group of Q heads); and a real incremental KV cache, which needs the FA kernel to accept `Nq != Nkv` — the actual prerequisite for CUDA graphs to show the kind of speedup they're capable of on real single-token decode. Next post.
