# Nsight Compute: tuned FlashAttention (kernel 4, v2) vs. PyTorch's real FA2

`kernels/flash_attention.py` (Triton, autotuned tile sizes, causal early-exit,
bf16 tensor-core `tl.dot`) vs. PyTorch's own FlashAttention-2 CUDA kernel
(`scaled_dot_product_attention` forced onto the `FLASH_ATTENTION` backend) on
the same L40S (`vllm-node-0`'s golden-snapshot image, run on a preemptible
node this session since on-demand L40S capacity was exhausted in this region
at the time), bf16, causal, `B=1, H=32, D=128`. Full headline benchmark table
is in `kernels/README.md`; this is the `ncu` deep-dive into what the
occupancy/bank-conflict numbers actually say about the remaining gap. Raw
output: `ncu_flash_attention_report.txt`.

`ncu` was run the same way kernel 3's was — see `matmul_ncu_report.md` (or
`rmsnorm_ncu_report.md`'s "Getting `ncu` to run at all" section) for the
host-Nsight-binary-bind-mounted-into-a-throwaway-container-with-`nvidia-dcgm`
-stopped recipe; unchanged here.

Both shapes profiled use **the same autotuned config** —
`BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3` — confirmed via
`TRITON_PRINT_AUTOTUNING=1` at both N=1024 and N=8192. At a real D=128 head
dim, `_prune_by_shared_mem` (see `flash_attention.py`) only leaves 3 bf16
configs standing, and the autotuner picked the largest one at every N tested,
not just these two — so this is one kernel/config's behavior across problem
sizes, not a per-shape retune.

## N=1024 (causal) — our weaker case (74.1% of FA2's TFLOPS in the wall-clock benchmark)

| metric | ours | SDPA-FA2 |
|---|--:|--:|
| kernel | `_flash_attn_fwd_kernel` | `flash_fwd_kernel<128,64,64,4,...>` |
| grid size (thread blocks) | 256 | **512** |
| block size (threads) | **256** | 128 |
| device time (`ncu` single-shot) | 186.69 us | 128.48 us |
| Tensor Core pipe active (% of peak) | 33.44% | **45.90%** |
| achieved occupancy (warps active) | **16.64%** | 14.92% |
| registers/thread | 201 | 184 |
| occupancy limiter (blocks/SM) | shared_mem: 1, registers: 1 | shared_mem: **2**, registers: **2** |
| shared-mem bank conflicts (ld / st) | **0** / 7,714 | 18,724 / 6,063 |
| DRAM throughput (% of 864 GB/s peak) | 17.63% | 25.53% |

## N=8192 (causal) — our strongest case (107.0% of FA2's TFLOPS — we're ahead)

| metric | ours | SDPA-FA2 |
|---|--:|--:|
| grid size (thread blocks) | 2,048 | **4,096** |
| block size (threads) | **256** | 128 |
| device time (`ncu` single-shot) | 6.26 ms | 4.65 ms |
| Tensor Core pipe active (% of peak) | 57.62% | **76.94%** |
| achieved occupancy (warps active) | **16.66%** | 16.38% |
| registers/thread | 201 | 184 |
| occupancy limiter (blocks/SM) | shared_mem: 1, registers: 1 | shared_mem: **2**, registers: **2** |
| shared-mem bank conflicts (ld / st) | **0** / 101,691 | 136,569 / 158,506 |
| DRAM throughput (% of 864 GB/s peak) | 5.18% | 7.03% |

Read the device-time row carefully, same caveat kernel 3's report already
flagged: `ncu`'s single-shot number (8 replay passes for counter collection)
disagrees with the wall-clock table in `kernels/README.md` at N=1024 (`ncu`
says FA2 wins here; the CUDA-event benchmark says we're at 74.1% of FA2 too,
consistent direction this time, unlike kernel 3's 8192³ case) — treat the
wall-clock numbers as the throughput claim, this table as occupancy/
utilization diagnostics.

## What the numbers actually say

**Occupancy is not the story here — it's a wash.** FA2's kernel fits 2
resident blocks/SM (both registers- and shared-memory-limited) against our
1, which sounds like FA2 should win occupancy outright. It doesn't, because
FA2's blocks are half the size (128 threads = 4 warps vs our 256 threads = 8
warps): 2 blocks x 4 warps and 1 block x 8 warps both land at 8 resident
warps/SM, and the measured `sm__warps_active` percentages come out
essentially tied (ours is a hair ahead at both sizes). Same lesson as kernel
3's matmul report: block *count* per SM and warps *resident* per SM aren't
the same number, and only the second one is occupancy.

**Tensor Core pipe utilization is where the real gap is**, and it's
substantial at both sizes (33.4% vs 45.9% at N=1024; 57.6% vs 76.9% at
N=8192) — this tracks the wall-clock gap far better than occupancy does.
`ncu` confirms both kernels run real HMMA instructions either way; the
difference is how much of the time those pipes are actually busy per
instruction issued, which comes down to instruction-level scheduling inside
the K/V loop (warp specialization, `cp.async`/`ldgsts` staging, MMA-fragment
layout) that a 7-config Triton autotune sweep over `BLOCK_M`/`BLOCK_N`/
`num_warps`/`num_stages` doesn't reach — the same "not cuBLAS/CUTLASS-level
engineering effort" gap kernel 3 already found against cuBLAS, showing up
again here against FA2's own hand-written CUTLASS kernel. It's also exactly
why we're *closer* to FA2 at N=8192 than N=1024: more total work amortizes a
fixed per-instruction efficiency gap better, the same shape kernel 3's
8192³ story took.

**Bank conflicts: zero on loads, real but not the bottleneck on stores.**
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` is exactly **0**
for our kernel at both sizes — the `offs_m[:, None]*stride + offs_d[None,
:]*stride` tile-load pattern for Q/K/V doesn't collide across banks, which
is the thing worth confirming rather than assuming. Store-side conflicts are
real and scale with the K/V loop's trip count (7,714 at N=1024's 8-tile
loop, 101,691 at N=8192's 64-tile loop — roughly linear in tile count, as
you'd expect for a per-iteration cost) — most likely `acc`'s running
rescale-and-accumulate write pattern each step. Two things keep this from
being the headline finding: FA2's own kernel has *more* total bank
conflicts (both ld and st, at both sizes) and still wins on wall-clock, so
conflicts alone don't explain the gap either; and chasing this further
would mean reading generated SASS to find the actual colliding access
pattern, which is real follow-up work, not something this pass did.

## Files

- `profile_flash_attention_ncu.py` — NVTX-scoped target script `ncu`
  launches (`ours` or `sdpa`, any shape/causal via CLI flags).
- `ncu_flash_attention_report.txt` — full raw `ncu` output (ours/SDPA-FA2 x
  N=1024/8192, causal).
- `benchmark_flash_attention_results.json` — the wall-clock CUDA-event
  measurements the headline table in `kernels/README.md` is built from.
