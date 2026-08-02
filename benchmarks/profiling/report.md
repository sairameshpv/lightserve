# GPU Profiling: Prefill vs Decode

Profiling of the vLLM server (`meta-llama/Meta-Llama-3-8B-Instruct`, single L40S GPU) to answer where GPU time goes during a single request's prefill step vs its decode steps. Two isolated probe requests were used throughout:

- **prefill-probe**: 140-token prompt (`long` category from `baseline_prompts.jsonl`), `max_tokens=1` — near-pure prefill, negligible decode.
- **decode-probe**: 2-token prompt (`"Hi"`), `max_tokens=200` — small prefill, 200 decode steps.

Two separate profiling sessions were run (nsys and torch.profiler can't share a process — both need CUPTI's single process-wide subscriber slot):

| | Session A (Nsight Systems) | Session B (PyTorch Profiler) |
|---|---|---|
| CUDA graphs | **On** (matches production/baseline config) | **Off** (`--enforce-eager` — required so op-level hooks aren't bypassed by graph replay) |
| Tool | `nsys` wrapping the container as PID 1, bind-mounted from the host's Nsight install | vLLM's built-in `--profiler-config.profiler torch` |
| Use | Authoritative "top 5 kernels by time" (below) | Cross-check + shows the cost of disabling CUDA graphs |

## Top 5 kernels by time (Session A, graph mode — production-representative)

**Prefill-probe** (140→1 tokens, total GPU-active time 1.92ms):

| % | Time | Instances | Kernel |
|---|---|---|---|
| 75.7% | 1.452ms | 1 | `gemvx::kernel` (LM-head vocab-logits projection) |
| 15.1% | 0.290ms | 32 | `flash_fwd_splitkv_kernel` (attention) |
| 4.0% | 0.077ms | 32 | `reshape_and_cache_flash_kernel` (KV-cache write) |
| 1.5% | 0.029ms | 1 | `cunn_SoftMaxForward` |
| 0.6% | 0.012ms | 4 | `_apply_write_kernel` |

**Decode-probe** (2→200 tokens, 200 steps, total GPU-active time 306.9ms):

| % | Time | Instances | Kernel |
|---|---|---|---|
| 94.6% | 290.4ms | 200 | `gemvx::kernel` (LM-head vocab-logits projection) |
| 1.9% | 5.75ms | 200 | `cunn_SoftMaxForward` |
| 0.8% | 2.49ms | 200 | `TopPSamplingFromProbKernel` |
| 0.3% | 1.03ms | 400 | `index_elementwise_kernel` |
| 0.2% | 0.70ms | 200 | `_temperature_kernel` |

Full derived data: `kernel_summary_prefill.csv`, `kernel_summary_decode.csv`.

## The main finding: LM-head projection dominates both phases equally

The **same kernel** — the vocab-logits GEMV (`gemvx::kernel`, computing `hidden_state @ lm_head_weight` over the full 128k-token vocabulary) — is the single largest cost in *both* prefill (75.7%) and decode (94.6%), at an almost perfectly constant **~1.45ms per generation step** (1.452ms for the one prefill step; 290.4ms / 200 = 1.452ms per decode step — essentially identical).

This makes sense once you consider what the op actually is: with batch size 1 (a single request, no concurrent load), computing logits is a **matrix-vector product**, not a matrix-matrix product — it reads the entire `hidden_dim × vocab_size` (4096 × ~128k) LM-head weight matrix from HBM once per token, regardless of how long the prompt was. It's memory-bandwidth-bound, not compute-bound, so prefill's extra attention/FFN work over 140 tokens barely moves the needle — attention is only 15.1% of the prefill window, and doesn't even make decode's top 5.

**Implication:** this cost should amortize away under real concurrent load (the baseline benchmarks ran at concurrency 50) — batching turns the GEMV into a genuine GEMM, which is compute-bound and scales far better with batch size than this single-request memory-bound probe shows. The 1-2 requests profiled here intentionally isolate a single phase; they are not a throughput measurement (see `benchmarks/README.md` for those numbers).

## Session B (eager mode): the cost of disabling CUDA graphs

Running the *same* two probes under `--enforce-eager` (required for torch.profiler's op-level hooks to see individual kernels instead of one opaque graph-replay node) tells a very different story — and quantifies what CUDA graphs are actually buying in production:

| | Session A (graphs on) | Session B (eager) | Ratio |
|---|---|---|---|
| Prefill total kernel time | 1.92ms | 26.74ms | **14x** |
| Decode total kernel time (200 steps) | 306.9ms | 4225.2ms | **14x** |

In eager mode, prefill's time is instead dominated by large GEMM kernels (`ampere_bf16_s16816gemm...` + `cutlass::Kernel2` ≈ 90.5% combined) rather than the LM-head GEMV — cuBLAS appears to select different (less graph-optimized) kernel variants when dispatched one Python op at a time. More strikingly, in eager mode the LM-head GEMV itself gets split into **~128 separate kernel launches per decode step** (vs. exactly 1 per step under graph replay), which is most of where the 14x decode slowdown comes from — CUDA graph capture is fusing/batching what would otherwise be a large number of small, launch-overhead-bound kernel dispatches into a single efficient replay.

**Caveat:** Session B's absolute numbers are not comparable to the production baseline benchmarks (graphs on) — they exist to isolate individual ops/kernels and to show this graph-mode-vs-eager gap, not to represent real serving latency.

## GPU utilization (`nvidia-smi dmon`, 1s sampling)

dmon's ~1s sampling floor means it's not meaningful for the sub-30ms prefill-probe window (at most 1 sample lands inside it). It's a good sanity check for the decode-probe though — GPU state during that window:

```
20:35:10   idle    (103W,   0% sm)
20:35:12   ramping (111W,  26% sm)   <- decode-probe starts
20:35:13   busy    (209W, 100% sm)
20:35:14   busy    (271W, 100% sm)
20:35:15   busy    (272W, 100% sm)
20:35:16   idle    (244W,   0% sm)   <- decode-probe ends
```

Confirms clean isolation — GPU goes from idle to pegged at 100% SM utilization exactly across the decode-probe's ~4.6s span and drops back to idle immediately after.

## Raw traces (not committed — see `.gitignore`)

Kept locally in `benchmarks/profiling/traces/` (gitignored, ~900MB total):
- `session_a_vllm.nsys-rep` — open in Nsight Systems GUI (`nsys-ui`) for the full interactive timeline.
- `torch_profiler/{prefill,decode}_rank0.pt.trace.json.gz` — open in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing` for the full-fidelity flame chart.
- `dmon.log`, `session_a_cuda_gpu_trace.csv` (full per-kernel timeline), `session_a_segmentation.json` (probe-window boundaries + how they were identified).

## Methodology notes

- Prefill vs decode isolation is **structural** (two differently-shaped requests), not via NVTX phase tagging — simpler, and unambiguous since each probe is (almost) purely one phase.
- The two probe windows inside Session A's single continuous trace were located by segmenting the kernel timeline on >50ms idle gaps, then disambiguated using the flash-attention kernel's grid dimension (`GrdX=3` for the 140-token prefill vs `GrdX=1` for the single-token decode/warmup requests) — see `session_a_segmentation.json`.
- vLLM 0.26.0 changed the profiler API from the older `VLLM_TORCH_PROFILER_DIR` env var to a `--profiler-config.profiler torch --profiler-config.torch_profiler_dir <path>` CLI flag; the env var is silently ignored with a warning in this version.
