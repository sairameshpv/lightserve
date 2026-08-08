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

## CI

`.github/workflows/kernels-ci.yml` runs `pytest kernels/tests/` on every push
touching `kernels/**`. GitHub-hosted runners have no GPU, so every
`requires_cuda`-marked case is skipped there — the CI job's job is to catch
import errors, shape-validation regressions, and packaging breakage, not to
re-verify numerics. Full correctness (the allclose-vs-PyTorch result above)
is verified by hand against a real GPU (see above); there is no self-hosted
GPU runner wired up yet.

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
