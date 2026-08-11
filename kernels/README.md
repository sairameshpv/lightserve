# Triton kernels

Hands-on Triton kernels, written to learn the block/tile programming model,
each one verified for correctness against a PyTorch reference and benchmarked
against the naive PyTorch equivalent.

## Kernel 1: fused bias-add + ReLU

`fused_bias_relu.py` computes `y = relu(x + bias)` for `x: [M, N]`,
`bias: [N]` (broadcast across rows) — the pattern that follows a
matmul/linear layer. The file itself is written as an annotated walkthrough
of Triton's tile/program model (program IDs → tile offsets → masking →
pointer arithmetic → load/compute/store); read it top to bottom.

This op is almost pure memory traffic (2 adds + 1 max per element, nothing
compute-heavy), so fusing buys nothing in FLOPs — the win is HBM round-trips.
Naive (`x + bias` then `relu(.)`) is 2 kernel launches: read x + read bias +
write tmp, then read tmp + write out. Fused is 1 launch: read x + read bias +
write out.

### Correctness

`tests/test_fused_bias_relu.py` checks the kernel against
`torch.relu(x + bias)` via `torch.testing.assert_close` (allclose, with
looser tolerance for bf16/fp16 than fp32) across a range of shapes —
including ones that aren't multiples of the 64×64 block size, to exercise
the boundary-masking path — plus a non-contiguous (transposed) input, an
all-negative sanity check on the ReLU half, and shape-mismatch validation.

Run for real on an L40S (`vllm-node-0`, torch 2.11.0+cu130, inside the
vllm-openai container):

```
$ python3 -m pytest kernels/tests/ -v
...
24 passed in 3.98s
```

All 24 cases (3 dtypes × 7 shapes + non-contiguous + all-negative + the
CPU-only shape check) pass.

### Benchmark

`benchmark_fused_bias_relu.py` times fused vs. naive via CUDA events
(median of 5×100-iter blocks) and re-verifies allclose immediately before
timing each shape, so a benchmark number is never reported for a kernel that
doesn't match the reference. Measured on the same L40S:

| shape | dtype | fused ms | naive ms | speedup | fused GB/s | naive GB/s |
|---|---|--:|--:|--:|--:|--:|
| 128×128 | fp32 | 0.0225 | 0.0147 | 0.65x | 5.8 | 17.9 |
| 128×128 | bf16 | 0.0217 | 0.0146 | 0.67x | 3.0 | 9.0 |
| 512×512 | fp32 | 0.0220 | 0.0143 | 0.65x | 95.4 | 293.5 |
| 512×512 | bf16 | 0.0220 | 0.0145 | 0.66x | 47.8 | 145.0 |
| 1024×1024 | fp32 | 0.0216 | 0.0148 | 0.68x | 388.0 | 1135.7 |
| 1024×1024 | bf16 | 0.0223 | 0.0144 | 0.65x | 188.4 | 582.8 |
| 4096×4096 | fp32 | 0.1829 | 0.3440 | **1.88x** | 733.8 | 780.4 |
| 4096×4096 | bf16 | 0.0219 | 0.0592 | **2.70x** | 3058.7 | 2267.3 |
| 4096×14336 | fp32 | 0.7751 | 1.4496 | 1.87x | 606.1 | 648.2 |
| 4096×14336 | bf16 | 0.4750 | 0.7177 | 1.51x | 494.6 | 654.6 |
| 8192×8192 | fp32 | 0.8847 | 1.6575 | 1.87x | 606.9 | 647.8 |
| 8192×8192 | bf16 | 0.5277 | 0.8203 | 1.55x | 508.7 | 654.5 |

Full data: `benchmark_results.json`.

**Findings:**

1. **Below ~1M elements, naive wins.** At 128×128 through 1024×1024, fused
   is actually *slower* (0.65–0.68x) despite doing half the HBM round-trips.
   Both fused and naive sit around 0.014–0.022ms regardless of shape here —
   dominated by fixed per-launch overhead (Triton's Python-side grid setup +
   kernel dispatch), not the tiny amount of actual data movement. Two
   already-warm cuBLAS/ATen elementwise kernels apparently dispatch faster
   than one Triton kernel at this scale. Same lesson as the roofline
   analysis in `../benchmarks/profiling/roofline/`: below some size, you're
   latency-bound, not bandwidth-bound, and neither kernel is anywhere near
   its ceiling.
2. **Fusion wins clearly once the tensor is HBM-traffic-bound** — from
   4096×4096 fp32 onward, fused is consistently ~1.5–1.9x faster, exactly the
   ~2x-fewer-round-trips you'd expect from the byte-counting argument above.
3. **bf16 4096×4096 is an outlier worth explaining, not a bug**: fused jumps
   to 3058.7 GB/s — over 3x the L40S's 864 GB/s HBM spec. At that dtype and
   shape the whole working set (x + out, ~33.5MB each) fits inside the
   L40S's L2 cache, so the benchmark's repeated-call timing loop is
   largely measuring L2-cache bandwidth, not HBM. The same shape at fp32
   (67MB per tensor, doesn't fit) drops straight back to ~734 GB/s, in line
   with real HBM throughput — confirming it's a cache-residency effect, not
   a measurement error.

## Kernel 2: fused residual-add + RMSNorm

`fused_rmsnorm_residual.py` computes, for `x`/`residual: [M, N]`,
`weight: [N]`:

```
h   = x + residual
out = h / sqrt(mean(h**2, dim=-1) + eps) * weight
```

returning `(out, h)` — `h` is the new residual, carried forward into the
next sublayer, matching the pre-norm decoder-layer boundary in every
LLaMA-family model. Unlike kernel 1's pure tile-parallel elementwise op,
RMSNorm needs a per-row *reduction* first, so the tiling strategy is
different: one Triton program per row, each loading and reducing its whole
row in one shot (`BLOCK_N = triton.next_power_of_2(N)`, masked). Variance is
accumulated in fp32 regardless of input dtype — same numerical-stability
convention LLaMA's own RMSNorm uses.

### Correctness

`tests/test_fused_rmsnorm_residual.py` checks both outputs against a
PyTorch-eager reference (fp32-accumulated variance, same as the kernel)
across LLaMA-real hidden sizes (4096, 8192) and non-power-of-2 widths, plus
a direct unit-RMS sanity check (`weight=1` ⇒ `mean(out**2) == 1` by
definition), a zero-row/eps-stability check, and rejection of shape
mismatches and non-contiguous inputs (the whole-row-at-stride-1 load
strategy requires contiguity, unlike kernel 1). Run for real on the same
L40S:

```
$ python3 -m pytest kernels/tests/ -v
...
46 passed in 4.63s
```

(24 from kernel 1 + 22 new: 3 dtypes × 6 shapes + 3 targeted checks + the
2 CPU-only validation tests.)

### Benchmark

`benchmark_fused_rmsnorm_residual.py`, same re-verify-then-time methodology
as kernel 1, across LLaMA-real hidden sizes (4096, 8192) and batch sizes
spanning decode (M=1) to prefill (M=2048):

| shape | dtype | fused ms | naive ms | speedup | fused GB/s | naive GB/s |
|---|---|--:|--:|--:|--:|--:|
| 1×4096 | fp32 | 0.0281 | 0.0582 | 2.07x | 2.3 | 2.3 |
| 1×4096 | bf16 | 0.0266 | 0.0916 | 3.44x | 1.2 | 1.0 |
| 8×4096 | fp32 | 0.0270 | 0.0577 | 2.14x | 19.4 | 18.2 |
| 8×4096 | bf16 | 0.0264 | 0.0917 | 3.48x | 9.9 | 7.9 |
| 128×4096 | fp32 | 0.0263 | 0.0579 | 2.20x | 319.1 | 289.7 |
| 128×4096 | bf16 | 0.0264 | 0.0923 | 3.50x | 159.1 | 125.0 |
| 2048×4096 | fp32 | 0.1923 | 0.2513 | 1.31x | 697.8 | 1068.1 |
| 2048×4096 | bf16 | 0.0268 | 0.2033 | **7.60x** | 2507.7 | 907.8 |
| 1×8192 | fp32 | 0.0264 | 0.0591 | 2.24x | 5.0 | 4.4 |
| 1×8192 | bf16 | 0.0267 | 0.0937 | 3.52x | 2.5 | 1.9 |
| 128×8192 | fp32 | 0.0264 | 0.0585 | 2.22x | 636.0 | 573.6 |
| 128×8192 | bf16 | 0.0264 | 0.0918 | 3.47x | 317.4 | 251.3 |
| 2048×8192 | fp32 | 0.4314 | 0.7921 | 1.84x | 622.2 | 677.8 |
| 2048×8192 | bf16 | 0.1916 | 0.8814 | 4.60x | 700.4 | 418.8 |

Full data: `benchmark_rmsnorm_results.json`.

**Unlike kernel 1, fused wins at every single shape/dtype tested here** —
1.3x to 7.6x, including the smallest (M=1, decode-shaped) cases. The
difference from kernel 1's small-shape story: naive add+RMSNorm launches
~7-11 separate kernels (add, upcast, pow, reduce, `+eps`, rsqrt, 2 multiplies,
downcast — see the Nsight Compute breakdown below), so its per-launch
overhead compounds far more than kernel 1's 2-launch naive path, enough to
outweigh fusion's overhead disadvantage even at M=1.

### Nsight Compute: memory bandwidth utilization vs. eager

Full writeup, including per-kernel `dram__throughput.avg.pct_of_peak_sustained_elapsed`
breakdown and how `ncu` was actually gotten to run in this environment (host
vs. container, DCGM's profiling-counter lock): **[`rmsnorm_ncu_report.md`](rmsnorm_ncu_report.md)**.
Headline, at M=2048×N=4096:

| dtype | path | kernels | achieved BW | % of 864 GB/s peak |
|---|---|--:|--:|--:|
| bf16 | **fused** | **1** | **743.5 GB/s** | **86.1%** |
| bf16 | naive | 11 | 693.7 GB/s | 80.3% |
| fp32 | **fused** | **1** | **787.2 GB/s** | **91.1%** |
| fp32 | naive | 7 | 764.4 GB/s | 88.5% |

The fused kernel reaches a *higher* fraction of peak HBM bandwidth than
naive's own blended average, not just less total traffic — naive's average
gets dragged down by tiny `[M,1]` reduction-output kernels (variance+eps,
rsqrt) that land at 0.5-0.6% of peak, pure latency-bound overhead fusion
sidesteps entirely by keeping that math in registers.

## Kernel 3: tiled GEMM (C = A @ B), tensor cores + shared memory

`tiled_matmul.py` computes `C = A @ B` for `A: [M, K]`, `B: [K, N]`, tiling
both operands into on-chip blocks that `tl.dot` reduces against — the payoff
here isn't fewer kernel launches (kernels 1 and 2's story), it's *reuse*:
every element loaded gets used for `BLOCK_M` or `BLOCK_N` multiply-adds
before it's evicted, instead of being re-streamed from HBM per output
element. Two things beyond kernels 1/2's model: `tl.dot` compiles to real
Tensor Core MMA instructions on this GPU (Ada) when operands are bf16/fp16,
and the grid uses an L2-locality "swizzle" (`GROUP_M`) so consecutive thread
blocks reuse the same handful of A-tiles instead of thrashing L2 with a
naive row-major tile order. 4 `@triton.autotune` configs, not an exhaustive
search — see `matmul_ncu_report.md` for what that leaves on the table
against cuBLAS.

### Correctness

`tests/test_tiled_matmul.py` checks against `torch.matmul` (cuBLAS) across
LLM-real shapes (square and FFN-rectangular), non-block-multiple boundary
cases on M/N *and* K, non-contiguous inputs, and dtype/shape-mismatch
rejection. Two real bugs turned up writing this kernel, both fixed and
worth naming: the K-loop's boundary mask only covered the K edge, not the
M/N edge, so edge tiles were reading past the tensor's actual allocation on
every K-step (silently discarded by the masked final store, but still
genuine undefined behavior); and `tl.dot` defaults to TF32 (truncated
mantissa) for fp32 operands while this PyTorch/driver's `torch.matmul`
fp32 default is strict IEEE, so the two weren't computing the same thing
until the kernel passed `input_precision="ieee"` explicitly. Run for real
on the same L40S:

```
$ python3 -m pytest kernels/tests/ -v
...
67 passed in 76.44s
```

(46 from kernels 1+2, 21 new: 3 dtypes × 6 shapes + non-contiguous + 2
CPU-only validation tests.)

### Benchmark: ours vs. cuBLAS

`benchmark_tiled_matmul.py`, bf16, 3 square sizes (the dtype/peak-spec every
other number in this repo is quoted against — see the roofline analysis).
CUDA-event wall-clock, median of 5×30-iter blocks, allclose-verified before
every timed size:

| size | ours | cuBLAS | efficiency (ours/cuBLAS) | ours, % of 362 TFLOPS peak |
|---|--:|--:|--:|--:|
| 1024³ | 50.2 TFLOPS | 152.8 TFLOPS | 32.8% | 13.9% |
| 4096³ | 212.5 TFLOPS | 238.9 TFLOPS | 88.9% | 58.7% |
| 8192³ | 186.2 TFLOPS | 161.3 TFLOPS | **115.4%** | 51.4% |

Full data: `benchmark_matmul_results.json`.

**cuBLAS does not win at every size tested here** — the honest result,
re-confirmed independently at 8192³ with a different seed (185.6 vs 169.2
TFLOPS). It wins decisively small, is nearly matched at 4096, and we're
reproducibly ~10-20% *ahead* at 8192. `ncu` (below) explains why: at 1024³,
cuBLAS's kernel spawns 2x more thread blocks than our autotuned tile size
does, which matters enormously there because 64 blocks can't fill the
L40S's 142 SMs regardless of per-block efficiency; by 8192³ there's enough
total work that this stops being the deciding factor, and a narrow 4-config
autotune search can land on a shape-specific answer cuBLAS's
heuristic-driven kernel selection didn't happen to pick for this exact size.

### Nsight Compute: why cuBLAS wins (and where it doesn't)

Full writeup with per-size `ncu` metrics (Tensor Core pipe utilization,
occupancy, register pressure, grid/block size) for both kernels:
**[`matmul_ncu_report.md`](matmul_ncu_report.md)**. Headline: at 1024³,
cuBLAS reaches 58.0% Tensor Core utilization against our 31.6% because its
kernel launches 128 thread blocks to our autotune-picked 64 — a
wave-quantization gap, not a "not using tensor cores" gap (`ncu` confirms
real HMMA instructions on both). At 8192³ the same block-count story
inverts what it predicts: cuBLAS still runs more blocks (4096 vs 2048) and
reaches higher Tensor Core utilization (96.9% vs 83.9%), yet our kernel
reaches 2x higher occupancy (16.7% vs 8.3%) and the two wall-clock
benchmarks above both measured us ahead regardless.

## Kernel 4: FlashAttention forward pass

`flash_attention.py` computes `O = softmax(Q @ K^T / sqrt(D)) @ V` for
`q, k, v: [B, H, N, D]`, without ever materializing the `[N, N]` score
matrix. One Triton program owns one query tile and sweeps the *entire* K/V
sequence for its `(batch, head)` sequentially — unlike kernel 3's grid
(every program independent), FlashAttention's inner loop carries state
(`m_i`/`l_i`/`acc`) from one K/V tile to the next via **online softmax**:
each step folds in a new tile, rescales the running output accumulator by
how much the running row-max moved (`alpha = exp(m_i - m_new)`), and only
normalizes once, after the loop, instead of needing the whole row up front
the way an untiled softmax would. Optional `causal` flag masks the upper
triangle. No backward pass (forward-only throughout).

`D` itself isn't tiled (`BLOCK_D = next_power_of_2(D)`, whole head dim
resident on-chip per tile, same whole-row approach as kernel 2's
`BLOCK_N`).

### v1: correctness, not speed

v1 (superseded by v2 below, kept here for the history) used fixed
`block_m=block_n=32` tiles, fp32 only, no `@triton.autotune`, and no causal
early-exit (masked the upper triangle but still swept the full K/V
sequence) — the only goal was getting the loop structure and
online-softmax accumulation right, allclose-verified against PyTorch's
SDPA, before touching anything performance-related. 32x32 was deliberately
conservative rather than tuned: one of this repo's earlier planning docs
estimated `Br=Bc=64` would fit the L40S's ~99KB/CTA shared-memory cap at a
real D=128 head dim with room to spare (~96KB by hand); that estimate
turned out optimistic on real hardware, underestimating Triton's own
load/compute pipelining overhead on top of the Q/K/V/O/S working set — see
below.

### Correctness

`tests/test_flash_attention.py` checks against
`torch.nn.functional.scaled_dot_product_attention` (SDPA) via
`torch.testing.assert_close` at fp32, across both `causal=False` and
`causal=True`, 6 `(B, H, N, D)` shapes spanning degenerate (`N=1`),
non-block-multiple `N`, a non-power-of-2 `D`, and real LLaMA-ish head dims
(64, 128), plus a non-contiguous (transposed-heads) input and
shape/dtype-mismatch rejection.

Two real bugs turned up running this on the L40S, both compile-time
failures Triton itself raised, not silent wrong answers:

1. **`tl.dot` requires its contraction dim ≥16.** The `D=8` test shape gave
   `BLOCK_D = next_power_of_2(8) = 8`, which is the K-dimension of the
   `Q @ K^T` dot and violates that minimum —
   `AssertionError: Input shapes should have M >= 1, N >= 1 and K >= 16`.
   Fixed by clamping `BLOCK_D = max(16, next_power_of_2(D))`.
2. **`BLOCK_M=BLOCK_N=64` at `D=128` exceeded shared memory**:
   `OutOfResources: Required: 180480, Hardware limit: 101376` (99KB/CTA on
   this Ada GPU). This is exactly the SRAM-budget question the loop
   structure was planned around, just with a real number attached — fixed
   by dropping the defaults to `block_m=block_n=32`, which fits at every
   `D` tested here. Bigger tiles are a real speed lever, bounded by this
   same cap, for a v2 `@triton.autotune` sweep.

Run for real on the same L40S (inside the vllm-openai container):

```
$ python3 -m pytest kernels/tests/test_flash_attention.py -v
...
15 passed in 12.57s
$ python3 -m pytest kernels/tests/ -v
...
82 passed in 84.91s
```

All 82 across kernels 1-4 pass — no regressions from the 67 kernels 1-3
already had.

No benchmark or `ncu` section for v1: it was scoped to correctness only,
per the plan. bf16/fp16 (tensor-core) dtypes, autotuned tile sizes, and a
causal early-exit were the natural v2 follow-ups — below.

### v2: autotuned tile sizes, causal early-exit, bf16 — tuned for speed

Same loop structure and online-softmax math as v1 (unchanged), three
additions on top:

1. **`@triton.autotune`** over `BLOCK_M`, `BLOCK_N`, `num_warps`,
   `num_stages` (7 configs, not exhaustive — same spirit as kernel 3's
   autotune).
2. **Causal early-exit**: the K/V loop now stops once past a query tile's
   diagonal (`hi = min(N, (pid_m+1)*BLOCK_M)`) instead of sweeping the full
   sequence and masking the upper triangle away after the fact — roughly
   halves FLOPs for causal attention, matching what FlashAttention-2 itself
   does and what a fair TFLOPS-vs-FA2 comparison requires.
3. **bf16** is the dtype these configs are actually tuned for (tensor-core
   `tl.dot`, fp32 accumulate) — fp32 still works (same dtype-generic
   kernel) but wasn't part of the tuning sweep.

**A real bug this surfaced**: `triton.autotune` does not skip a config that
fails to compile — it raises `OutOfResources` and crashes the *entire*
search the first time it hits one, same failure mode v1 hit at its single
fixed tile size, just now inside a search over 7. Since these configs were
picked for bf16, most of them don't fit an fp32 call at all once
`BLOCK_D` (`next_power_of_2(D)`, clamped ≥16) exceeds 64 — fp32 tiles are
2x the bytes of bf16 (the same v1 lesson, again, at bigger tiles). Rather
than hand-estimate a second time (v1's hand-estimate already undercounted
Triton's real pipelining overhead once), every `(config, dtype, BLOCK_D)`
combination was actually compiled on the L40S to get ground truth, which is
now baked into `_prune_by_shared_mem` (a `prune_configs_by={"early_config_prune":
...}` hook) — e.g. at a real `D=128` bf16 call, only 3 of the 7 configs fit
at all; at fp32/`D=128`, only the dedicated small safety-net config
(`BLOCK_M=BLOCK_N=32`) does. Without this, the fp32 correctness tests at
`D=96`/`128` fail outright with the same `OutOfResources` error, not a
slow-but-working result.

#### Correctness (v2)

`tests/test_flash_attention.py` now also parametrizes over `dtype` (fp32,
bf16) on top of v1's shapes/causal matrix — 27 cases total (was 15).
Run for real (on a preemptible L40S node this session — see below):

```
$ python3 -m pytest kernels/tests/test_flash_attention.py -v
...
27 passed in 30.69s
$ python3 -m pytest kernels/tests/ -v
...
94 passed in 102.53s
```

All 94 across kernels 1-4 pass — no regressions from the 67 kernels 1-3 had.

#### Benchmark: ours vs. PyTorch's real FlashAttention-2

`benchmark_flash_attention.py` compares against
`scaled_dot_product_attention` forced onto the `FLASH_ATTENTION` backend via
`torch.nn.attention.sdpa_kernel` — this *is* FA2 (whatever version this
torch build vendors), not an approximation of it, same "ours vs. the real
thing" shape as kernel 3's ours-vs-cuBLAS benchmark. bf16, `B=1, H=32,
D=128` (LLaMA-3-8B's real head count/dim), CUDA-event wall-clock, median of
5×30-iter blocks, allclose-verified before every timed shape:

| shape (B,H,N,D) | causal | ours TFLOPS | FA2 TFLOPS | % of FA2 | % of 362 TFLOPS peak |
|---|---|--:|--:|--:|--:|
| 1x32x512x128 | False | 80.6 | 97.1 | 83.0% | 22.3% |
| 1x32x512x128 | True | 40.9 | 47.7 | 85.7% | 11.3% |
| 1x32x1024x128 | False | 172.4 | 209.2 | 82.4% | 47.6% |
| 1x32x1024x128 | True | 102.0 | 137.6 | 74.1% | 28.2% |
| 1x32x2048x128 | False | 181.8 | 215.1 | 84.5% | 50.2% |
| 1x32x2048x128 | True | 125.6 | 173.1 | 72.6% | 34.7% |
| 1x32x4096x128 | False | 177.8 | 183.1 | 97.1% | 49.1% |
| 1x32x4096x128 | True | 146.2 | 146.8 | **99.6%** | 40.4% |
| 1x32x8192x128 | False | 184.0 | 193.6 | 95.1% | 50.8% |
| 1x32x8192x128 | True | 166.6 | 155.7 | **107.0%** | 46.0% |

Full data: `benchmark_flash_attention_results.json`.

**Every shape clears the >=60% of FA2 target, by a wide margin** — the
worst case (N=1024, causal) is still at 74.1%, and by N=4096/8192 we're at
parity-or-ahead of FA2 (99.6% and 107.0% causal). Same shape-dependent
story as kernel 3 vs cuBLAS: FA2's more numerous, hand-tuned configs (and a
real CUTLASS-hand-written inner loop, not a 7-config Triton sweep) win
decisively at small problem sizes where per-block/per-launch overhead
dominates; more total work amortizes that gap, and by 8192 tokens we're
reproducibly ahead.

#### Nsight Compute: occupancy and shared-memory bank conflicts

Full writeup with per-size `ncu` metrics (Tensor Core utilization,
occupancy, register pressure, shared-memory bank conflicts) for both
kernels: **[`ncu_flash_attention_report.md`](ncu_flash_attention_report.md)**.
Headline: **occupancy is a wash, not the story** — FA2 fits 2 resident
blocks/SM against our 1 (both register- and shared-memory-limited), but
FA2's blocks are half the threads (128 vs our 256), so both land at ~8
resident warps/SM either way, and measured `sm__warps_active` comes out
essentially tied (ours is a hair ahead at both N=1024 and N=8192). The real
gap is **Tensor Core pipe utilization** (33.4% vs 45.9% at N=1024, 57.6% vs
76.9% at N=8192) — FA2's hand-written CUTLASS inner loop schedules
tensor-core instructions more efficiently per cycle than a 7-config Triton
autotune sweep reaches, the same kind of engineering-effort gap kernel 3
found against cuBLAS. **Bank conflicts**: `ncu` confirms **zero** shared-
memory load conflicts for our kernel at both sizes (the Q/K/V tile-load
pointer arithmetic doesn't collide across banks) — store-side conflicts are
real and scale with the K/V loop's trip count, but FA2's kernel has *more*
total bank conflicts (both load and store) and still wins on wall-clock, so
conflicts aren't the bottleneck either; root-causing the store conflicts
further would mean reading generated SASS, flagged as real follow-up, not
done here.

### Backward pass (recomputation strategy)

`flash_attention_backward.py` adds a backward pass on top of the forward
kernel above -- gradients w.r.t. Q, K, V, computed without ever
materializing the `[N, N]` score/probability matrix either. "Recomputation
strategy" (the FlashAttention paper's term): the forward pass saves only
`O` and one extra scalar per row (`L`, that row's log-sum-exp), and the
backward pass recomputes each `S_ij`/`P_ij` tile from `Q`/`K` on the fly
using that saved scalar, instead of ever having stored the full matrix.
Standard FlashAttention-2 backward algorithm: one program per K/V tile,
sweeping every Q tile, `dK`/`dV` accumulated locally and stored once,
`dQ` accumulated across programs via `tl.atomic_add` into a float32
buffer. Full derivation and scope notes (plain MHA, no causal early-exit,
fixed non-autotuned tile size) are in that file's module docstring.

#### Correctness

`tests/test_flash_attention_backward.py`: `torch.autograd.gradcheck`
(numerical vs. analytical gradients, float32, independent of any reference
attention implementation) plus dQ/dK/dV vs. PyTorch's own autograd through
SDPA (fp32 + bf16, LLM-real shapes) -- same "two independent verification
methods" habit as the forward kernel's own test file.

**A real bug turned up running this on the L40S**: the backward kernel
keeps more tiles resident per program (K, V, plus each step's Q, dO -- 4
big tiles) than the forward-only kernel (Q resident + streamed K, V --
effectively 3), so the 32x32 tile size proven safe for fp32/D=128 in this
file's own v1 section above wasn't safe here --
`OutOfResources: Required 107008, Hardware limit 101376` at fp32/D=128.
Fixed with a dtype/head-dim-aware tile size (`_pick_block_sizes`: 16x16
for fp32 at D>64, 32x32 otherwise) instead of one fixed constant.

```
$ python3 -m pytest kernels/tests/test_flash_attention_backward.py -v
...
23 passed in 32.52s   # includes all 4 test_gradcheck cases
```

#### Benchmark: ours vs. PyTorch's real FlashAttention-2 backward

`benchmark_flash_attention_backward.py`, same FA2-forcing method as the
forward benchmark, extended to `.backward()`. Unlike the forward kernel's
v2, this backward kernel is still "v1" by its own docstring -- fixed tile
size, no `@triton.autotune` -- so a real gap against FA2's tuned backward
is the expected result here, not a bug:

| shape (B,H,N,D) | causal | ours TFLOPS | FA2 TFLOPS | % of FA2 | % of 362 TFLOPS peak |
|---|---|--:|--:|--:|--:|
| 1x32x512x128 | False | 14.3 | 90.5 | 15.8% | 3.9% |
| 1x32x512x128 | True | 11.4 | 44.5 | 25.5% | 3.1% |
| 1x32x1024x128 | False | 15.4 | 137.2 | 11.2% | 4.2% |
| 1x32x1024x128 | True | 13.9 | 100.2 | 13.8% | 3.8% |
| 1x32x2048x128 | False | 16.0 | 142.6 | 11.2% | 4.4% |
| 1x32x2048x128 | True | 15.0 | 122.4 | 12.2% | 4.1% |
| 1x32x4096x128 | False | 16.2 | 157.7 | 10.2% | 4.5% |
| 1x32x4096x128 | True | 15.6 | 138.1 | 11.3% | 4.3% |

Gradients verified allclose against SDPA-flash before every timed pair.
Full data: `benchmark_flash_attention_backward_results.json`.

**Ours plateaus around 14-16 TFLOPS regardless of N; FA2 scales up to
157.7** -- exactly what a fixed-tile, non-autotuned kernel looks like next
to a tuned one: no tile-size lever to pull as the problem grows, so it
stays flat while FA2's larger effective tiles and better pipelining keep
extracting more throughput at bigger N. This is the same gap kernel 4's
own *forward* v1 would have shown against FA2 had it been benchmarked
(it wasn't -- v1 was scoped to correctness only, see above); an autotuned
backward v2 (bigger tiles, `@triton.autotune` over them, possibly a causal
early-exit on the K/V-tile loop bound) is the natural follow-up, not done
here.

### A capacity note, unrelated to the kernel

This tuning session ran on a **preemptible** L40S node, not the on-demand
`vllm-node-0` used for kernels 1-3 and v1: on-demand `gpu-l40s-a` capacity
in `eu-north1` hit tenant-wide `LIMIT_REACHED` (confirmed via `nebius
capacity resource-advice list`, not just a retry-and-hope guess) at the
time. `terraform/variables.tf` now has a `preemptible` variable
(`terraform.tfvars` has it set to `true` with a comment to flip back once
capacity frees up) for next time this happens.

## CI

`.github/workflows/kernels-ci.yml` runs `pytest kernels/tests/ model/tests/`
on every push touching `kernels/**` or `model/**`. GitHub-hosted runners
have no GPU, so every `requires_cuda`-marked case is skipped there — the CI
job's job is to catch import errors, shape-validation regressions, and
packaging breakage, not to re-verify numerics. Full correctness (the
allclose-vs-PyTorch result above) is verified by hand against a real GPU
(see above); there is no self-hosted GPU runner wired up yet.

## Files

- `fused_bias_relu.py` — kernel 1, with inline commentary on the block/tile
  model.
- `tests/test_fused_bias_relu.py` — kernel 1 correctness tests (allclose vs.
  `torch.relu(x + bias)`).
- `benchmark_fused_bias_relu.py` — kernel 1 fused vs. naive timing +
  bandwidth, re-verifies allclose before every timed shape.
- `benchmark_results.json` — kernel 1 raw output from the L40S run above.
- `fused_rmsnorm_residual.py` — kernel 2, with inline commentary on the
  row-reduction tiling strategy.
- `tests/test_fused_rmsnorm_residual.py` — kernel 2 correctness tests.
- `benchmark_fused_rmsnorm_residual.py` — kernel 2 fused vs. naive timing +
  bandwidth.
- `benchmark_rmsnorm_results.json` — kernel 2 raw benchmark output.
- `profile_rmsnorm_ncu.py` / `ncu_rmsnorm_report.txt` /
  `rmsnorm_ncu_report.md` — kernel 2's Nsight Compute profiling target,
  raw output, and writeup.
- `tiled_matmul.py` — kernel 3, with inline commentary on tiling-for-reuse
  and the L2-locality grid swizzle.
- `tests/test_tiled_matmul.py` — kernel 3 correctness tests (allclose vs.
  `torch.matmul`/cuBLAS).
- `benchmark_tiled_matmul.py` — kernel 3 ours-vs-cuBLAS TFLOPS + efficiency
  %, at 3 square sizes.
- `benchmark_matmul_results.json` — kernel 3 raw benchmark output.
- `profile_matmul_ncu.py` / `ncu_matmul_report.txt` / `matmul_ncu_report.md`
  — kernel 3's Nsight Compute profiling target, raw output, and the
  why-cuBLAS-wins (and doesn't) writeup.
- `flash_attention.py` — kernel 4, with inline commentary on the
  online-softmax loop structure (v1) and the autotune/causal-early-exit/
  bf16 tuning pass (v2).
- `tests/test_flash_attention.py` — kernel 4 correctness tests (allclose
  vs. `torch.nn.functional.scaled_dot_product_attention`, causal +
  non-causal, fp32 + bf16).
- `benchmark_flash_attention.py` — kernel 4 v2 ours-vs-PyTorch's-real-FA2
  TFLOPS + % of FA2, across 5 sequence lengths x causal/non-causal.
- `benchmark_flash_attention_results.json` — kernel 4 v2 raw benchmark
  output.
- `profile_flash_attention_ncu.py` / `ncu_flash_attention_report.txt` /
  `ncu_flash_attention_report.md` — kernel 4 v2's Nsight Compute profiling
  target, raw output, and the occupancy/bank-conflict writeup.
- `flash_attention_backward.py` — kernel 4b, the backward pass
  (recomputation strategy) built on top of kernel 4's forward.
- `tests/test_flash_attention_backward.py` — kernel 4b correctness tests
  (`torch.autograd.gradcheck` + dQ/dK/dV vs. PyTorch's autograd through
  SDPA).
- `benchmark_flash_attention_backward.py` — kernel 4b ours-vs-PyTorch's-
  real-FA2-backward TFLOPS + % of FA2, across 4 sequence lengths x
  causal/non-causal.
- `benchmark_flash_attention_backward_results.json` — kernel 4b raw
  benchmark output.
