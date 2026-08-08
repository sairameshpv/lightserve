# Nsight Compute: tiled GEMM (kernel 3) vs. cuBLAS

`kernels/tiled_matmul.py` (Triton, tensor-core `tl.dot`, `GROUP_M`-swizzled
grid for L2 reuse) vs. `torch.matmul` (cuBLAS/cuBLASLt) on the same L40S
(`vllm-node-0`), bf16, three square sizes. Full headline table and findings
are in `kernels/README.md`; this doc is the `ncu` deep-dive into *why* the
gap looks the way it does. Raw output: `ncu_matmul_report.txt`.

## Headline (CUDA-event wall-clock, median of 5×30-iter blocks)

| size | ours | cuBLAS | efficiency (ours/cuBLAS) | ours, % of 362 TFLOPS peak |
|---|--:|--:|--:|--:|
| 1024³ | 50.2 TFLOPS | 152.8 TFLOPS | 32.8% | 13.9% |
| 4096³ | 212.5 TFLOPS | 238.9 TFLOPS | 88.9% | 58.7% |
| 8192³ | 186.2 TFLOPS | 161.3 TFLOPS | **115.4%** | 51.4% |

Re-confirmed the 8192³ result independently (separate script, different
random seed): 185.6 vs 169.2 TFLOPS, ours still ahead. **cuBLAS does not
win at every size** — it wins decisively small, is nearly matched at
4096, and we're reproducibly ~10-20% ahead at 8192. That's the real,
three-part story; "cuBLAS always wins" would have been a tidier headline
but isn't what the hardware says.

## Why cuBLAS wins big at 1024³ (`ncu`, both kernels profiled in isolation via NVTX)

| metric | ours | cuBLAS |
|---|--:|--:|
| kernel | `_matmul_kernel` | `cutlass_..._128x64_32x6_nn_align8` |
| grid size (thread blocks) | **64** | **128** |
| block size (threads) | 128 | 128 |
| device time (`ncu` single-shot) | 43.97 us | 23.97 us |
| Tensor Core pipe active (% of peak) | **31.6%** | **58.0%** |
| achieved occupancy (warps active) | 8.3% | 8.3% |
| registers/thread | 254 | 142 |
| occupancy limiter (blocks/SM) | registers: 2 | registers: 3 |

Triton's autotuner (searching `tiled_matmul.py`'s 4 configs) picked
`BLOCK_M=128, BLOCK_N=128` for this shape, producing **64** total thread
blocks (`1024/128 × 1024/128`). cuBLAS's chosen CUTLASS kernel
(`128x64` tile) produces **128** — twice as many. The L40S has 142 SMs;
with only 64 blocks, well over half the chip has literally nothing to run
regardless of how efficient each block is. Both kernels land at the *same*
8.3% achieved-occupancy-per-resident-block, so this isn't a per-SM
occupancy story — it's a **wave-quantization** story: cuBLAS's finer tiling
keeps more SMs simultaneously busy, which is exactly what shows up as
its 58.0% vs our 31.6% Tensor Core pipe utilization. Our kernel's larger
128×128 accumulator tile also costs nearly 2x the registers/thread (254 vs
142) for the fp32 accumulator alone, tightening the occupancy ceiling
further (limited to 2 resident blocks/SM vs cuBLAS's 3) — compounding, not
causing, the gap.

None of this is about tensor cores being used or not — `sm__pipe_tensor_op_hmma_cycles_active`
confirms both kernels run real HMMA instructions. It's that our autotune
search only has 4 candidate tile shapes, none finely-grained enough for a
problem this small, while cuBLAS effectively already ran that search (once,
offline, per architecture) across a far larger space and cached the answer
as a heuristic lookup.

## Why we're competitive-to-ahead at 8192³

| metric | ours | cuBLAS |
|---|--:|--:|
| kernel | `_matmul_kernel` (BLOCK 128x256x64) | `cutlass_..._128x128_32x4_nn_align8` |
| grid size (thread blocks) | 2048 | **4096** |
| block size (threads) | 256 | 128 |
| device time (`ncu` single-shot) | 8.46 ms | 7.32 ms |
| Tensor Core pipe active (% of peak) | 83.9% | 96.9% |
| achieved occupancy (warps active) | **16.7%** | 8.3% |
| registers/thread | 233 | 224 |
| occupancy limiter (blocks/SM) | registers: 1 | registers: 2 |

Read the `ncu` device-time row carefully: **it disagrees with the wall-clock
table above** (cuBLAS looks faster here, 7.32ms vs 8.46ms) even though two
independent repeated-timing runs both measured *us* ahead by 10-20%. This
is the same kind of wall-clock-vs-`ncu`-single-shot divergence kernel 2's
report ran into at fp32 — `ncu`'s number is one profiled call under 8 replay
passes for counter collection, not the steady-state median a real caller
experiences, and at multi-millisecond kernel durations, GPU clock/power
state between separate profiled launches is a real confound. Treat the
wall-clock table as the throughput claim and this table as occupancy/
utilization diagnostics only, not a second throughput measurement --
the two aren't measuring the same thing.

What's structurally solid regardless of which duration number you trust:
cuBLAS launches **2x more thread blocks** (4096 vs 2048) at this size too,
and reaches higher Tensor Core utilization (96.9% vs 83.9%) doing it — it's
still, in some sense, "better" per this metric. But our kernel achieves
**2x higher occupancy** (16.7% vs 8.3% warps-active) with half the blocks,
because at this size there's finally enough total work (2048 blocks across
142 SMs is ~14 waves) that our coarser tiling no longer starves the chip
the way it did at 1024 -- both kernels are working the GPU hard, just via
different points on the tile-size/block-count tradeoff, and ours happens to
land on the faster side of it for this specific shape. This is consistent
with the general, well-known fact that cuBLAS's heuristic algorithm
*selection* (picking from its precompiled kernel library) isn't a perfect,
exhaustive-search oracle for every possible shape — it's tuned to be very
good on average, not optimal on every input, and a narrow autotune search
can occasionally land on a better answer for one specific shape it evaluated
directly that cuBLAS's heuristic didn't.

## A smaller finding along the way

cuBLAS's kernel name at 1024³ is
`cutlass_80_tensorop_bf16_s16816gemm_**relu**_bf16_128x64_32x6_nn_align8` —
it includes a fused ReLU epilogue in its *name*, despite `torch.matmul(a, b)`
never asking for one. Confirmed via allclose that the actual returned values
are NOT relu'd (a plain matmul on Gaussian inputs is ~50% negative; ours
matched cuBLAS's output within tolerance, so cuBLAS's output has real
negative values too). This is CUTLASS's kernel-library convention: ship one
epilogue-fusable kernel template family and select/parameterize it at
runtime (relu disabled here), rather than compile a separate kernel per
possible epilogue -- cuBLASLt happened to reuse a GEMM+optional-ReLU kernel
for a plain GEMM call because it was the best match for this shape either
way.

## Files

- `profile_matmul_ncu.py` — NVTX-scoped target script `ncu` launches.
- `ncu_matmul_report.txt` — full raw `ncu` output (ours/cuBLAS x 1024/8192³).
- `benchmark_matmul_results.json` — the wall-clock CUDA-event measurements
  the headline table above is built from.
