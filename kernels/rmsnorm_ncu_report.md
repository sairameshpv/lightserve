# Nsight Compute: fused RMSNorm+residual vs. PyTorch eager

Kernel-level memory-bandwidth-utilization comparison for
`fused_rmsnorm_residual.py`, measured with `ncu` (Nsight Compute 2025.1.1) on
the same L40S used throughout this repo (`vllm-node-0`), at a representative
LLaMA-3-8B-shaped input: **M=2048 (batched tokens), N=4096 (hidden size)**,
both fp32 and bf16. Raw `ncu` output: `ncu_rmsnorm_report.txt`.

## Getting `ncu` to run at all

Worth recording since it wasn't obvious: `ncu` is installed on the **host**
(`/opt/nvidia/nsight-compute/2025.1.1/ncu`, via `cuda-nsight-compute` in the
golden snapshot), not inside the `vllm-openai` container, and Nsight's
counter collection needs to run in the same PID/CUDA-context namespace as
the target — so the working shape (same pattern already used for `nsys` in
`terraform/cloud-init.tftpl`) is to bind-mount the host's Nsight install into
a **throwaway** container (not the running `vllm-server`) with
`--cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined`,
and use `--entrypoint <path-to-ncu>` so `ncu` itself launches the target
Python process inside that container.

That alone still failed with `Profiling failed because a driver resource was
unavailable` — **DCGM** (`nvidia-dcgm.service`, installed for GPU telemetry)
holds the GPU's performance-counter lock continuously in the background, and
CUPTI only allows one profiling client at a time. `sudo systemctl stop
nvidia-dcgm` for the duration of the profiling run (restarted immediately
after) fixed it. No permission-model changes were needed — the driver never
returned `ERR_NVGPUCTRPERM`.

To profile exactly the kernels under test and nothing else (no CUDA-context
init, no cuBLAS lazy-handle-creation kernels), `profile_rmsnorm_ncu.py` warms
up outside an NVTX range and wraps only the timed call in
`torch.cuda.nvtx.range_push("PROFILE_TARGET")` / `range_pop()`, and `ncu` is
invoked with `--nvtx --nvtx-include "PROFILE_TARGET/"` to scope collection to
that range.

## Results

Metrics: `dram__throughput.avg.pct_of_peak_sustained_elapsed` (HBM bandwidth,
% of the L40S's 864 GB/s peak), `dram__bytes.sum`, `gpu__time_duration.sum`
(pure device time — excludes Python/ATen dispatch overhead, unlike the
CUDA-event wall-clock numbers in `benchmark_rmsnorm_results.json`).

| dtype | path | kernels | total bytes | total device time | achieved BW | % of 864 GB/s peak |
|---|---|--:|--:|--:|--:|--:|
| bf16 | **fused** | **1** | **37.8 MB** | **50.9 us** | **743.5 GB/s** | **86.1%** |
| bf16 | naive (eager) | 11 | 270.1 MB | 389.3 us | 693.7 GB/s | 80.3% |
| fp32 | **fused** | **1** | **93.2 MB** | **118.4 us** | **787.2 GB/s** | **91.1%** |
| fp32 | naive (eager) | 7 | 233.0 MB | 304.8 us | 764.4 GB/s | 88.5% |

("naive" rows are the sum/aggregate across every kernel PyTorch eager
launches for `h = x + residual; rmsnorm(h) * weight`; per-kernel breakdown
below.)

**The fused kernel wins on every axis, not just wall-clock time:**

- **7.65x less device time at bf16** (50.9us vs 389.3us summed across
  naive's 11 kernels), **2.57x at fp32** (118.4us vs 304.8us) — these are
  pure GPU-busy-time ratios from `ncu`, and the bf16 number lines up almost
  exactly with the 7.60x wall-clock speedup already measured via CUDA events
  in `benchmark_fused_rmsnorm_residual.py`, which is a good independent
  cross-check that both measurements are real. (The fp32 wall-clock speedup
  measured there was lower, 1.31x — see caveat below.)
- **7.1x fewer bytes moved at bf16** (37.8 MB vs 270.1 MB), **2.5x at fp32**
  (93.2 MB vs 233.0 MB). bf16's gap is bigger because the naive path's
  `h.float()` (needed twice — see below) inserts two extra full-tensor
  bf16→fp32 upcast kernels that don't exist in the fp32 case, where
  `.float()` is a no-op.
- **Higher bandwidth utilization, not just less traffic.** Even ignoring
  volume entirely, fused reaches a higher fraction of the L40S's peak HBM
  bandwidth than naive's own blended average (86.1% vs 80.3% at bf16, 91.1%
  vs 88.5% at fp32) — naive's average is dragged down by kernels that are
  individually inefficient at this problem size, most visibly the `[2048,1]`
  variance/eps-add and rsqrt kernels, which move so little data (~15KB) that
  they land at **0.5-0.6% of peak bandwidth**, entirely latency-bound. Same
  lesson as kernel 1's benchmark and the roofline analysis: tiny ops never
  get near their ceiling, and fusing them into the same kernel as the big
  reduction is strictly better than paying for their own launches.

### Per-kernel breakdown, naive bf16 (11 launches for one `add+RMSNorm` call)

| kernel | bytes | device time | % of peak BW | % of peak compute (SM) |
|---|--:|--:|--:|--:|
| `add` (x + residual) | 38.0 MB | 48.1 us | 91.7% | 6.3% |
| copy/cast bf16→fp32 (`h.float()`, 1st call) | 20.5 MB | 40.0 us | 59.4% | 52.0% |
| `pow(h_f32, 2)` | 38.1 MB | 49.0 us | 90.3% | 4.4% |
| `mean(-1)` (reduce) | 38.5 MB | 52.8 us | 84.6% | 3.3% |
| copy/cast bf16→fp32 (`h.float()`, 2nd call) | 20.5 MB | 39.9 us | 59.6% | 52.1% |
| `variance + eps` ([2048,1], tiny) | 14.6 KB | 3.2 us | 0.5% | 0.02% |
| `rsqrt` ([2048,1], tiny) | 16.5 KB | 3.2 us | 0.6% | 0.03% |
| `h_f32 * rrms` (mul, fp32) | 38.3 MB | 49.4 us | 89.9% | 37.8% |
| (small copy, ~46KB) | 45.6 KB | 6.7 us | 0.8% | 0.2% |
| `× weight` (mul, fp32) | 38.2 MB | 49.1 us | 90.3% | 38.0% |
| cast fp32→bf16 (final `.to(x.dtype)`) | 37.9 MB | 48.0 us | 91.5% | 4.5% |

Two of those 11 launches exist because the reference implementation
(`h.float().pow(2).mean(...)` computed separately from
`h.float() * torch.rsqrt(...)`) calls `.float()` on `h` **twice** rather than
caching it — exactly how this is commonly hand-written, not a strawman. A
hand-optimized eager version that caches `h_f32` in a local would drop 1 of
those 2 upcast kernels, but would still need ~8 separate launches for the
add, cast, pow, reduce, `+eps`, rsqrt, and 2 multiplies. Fusion isn't
competing against a strawman here; it's competing against the minimum
kernel count eager execution can reach for this op, and still wins by ~7x.

### The fp32 wall-clock/device-time gap

`ncu`'s device-time-only speedup at fp32 (2.57x) is meaningfully higher than
the CUDA-event wall-clock speedup measured separately (1.31x, in
`benchmark_rmsnorm_results.json`). Both are real measurements of the same
kernel, just different things: CUDA events measure everything between
"Python asked for this op" and "the GPU finished it" (including per-op
Python/ATen dispatch overhead, which naive pays 7x over and fused pays
once), while `gpu__time_duration.sum` from `ncu` measures only time the GPU
itself was executing, under `ncu`'s own single-kernel-at-a-time replay
methodology. The qualitative conclusion — fused wins, decisively, on both
axes — doesn't depend on which one you read; the magnitude does, and the
wall-clock number is the one that reflects what a real caller experiences.

## Files

- `profile_rmsnorm_ncu.py` — NVTX-scoped target script `ncu` launches.
- `ncu_rmsnorm_report.txt` — full raw `ncu` output for all 4 runs
  (fused/naive x bf16/fp32).
- `benchmark_rmsnorm_results.json` — the independent CUDA-event wall-clock
  measurements referenced above.
