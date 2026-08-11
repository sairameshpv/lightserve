# Minimal LLaMA forward pass + CUDA Graph decode

Two pieces, both built on `kernels/`:

1. **`minimal_llama.py`** — a LLaMA-shaped decoder-only transformer forward
   pass, wired to this repo's own Triton kernels
   (`kernels/tiled_matmul.py`'s `matmul` for every Linear,
   `kernels/fused_rmsnorm_residual.py`'s `fused_add_rmsnorm` for every
   pre-norm, `kernels/flash_attention.py`'s `flash_attention_forward` for
   every attention call) instead of `nn.Linear`/`F.rms_norm`/
   `F.scaled_dot_product_attention`.
2. **`cuda_graph_decode.py`** — captures one autoregressive decode step
   (a full call into (1)) as a `torch.cuda.CUDAGraph`, and benchmarks it
   against the same decode loop run eager.

Scope decisions (plain multi-head attention not LLaMA-3's real GQA, RoPE
and the MLP's SiLU-gate left as plain PyTorch, random not real-checkpoint
weights) are documented in `minimal_llama.py`'s module docstring; the
static-full-buffer decode design (why every step reruns the whole
`max_seq_len` self-attention instead of an incremental KV cache) is in
`cuda_graph_decode.py`'s.

## Correctness

`tests/test_minimal_llama.py` checks the kernel-built forward against
`reference_llama_forward` (same weights, plain PyTorch throughout —
`F.linear`, `F.scaled_dot_product_attention`, a hand-written RMSNorm), fp32
+ bf16, causal + non-causal, 3 shapes including a non-block-multiple `N`.

**A real bug turned up running this on the L40S**: each layer's MLP output
was only ever folded into the residual stream via the *next* layer's
`fused_add_rmsnorm` call (its `h = x + residual` add at the top of the
loop) — correct for layers 1..n-1, but the *last* layer has no next
iteration to do that fold, and the final-norm call was written to pass
`torch.zeros_like(x)` instead of the real last-layer `x`, silently
dropping the final MLP sublayer's output entirely. Caught immediately by
this test file (96.8% of elements mismatched on a 1-layer model, not a
close-but-off tolerance failure) — fixed by feeding the real `x` into the
final `fused_add_rmsnorm` call instead. See `minimal_llama.py`'s
`llama_forward` for the fix and the comment explaining why the zero was
wrong.

**A second real bug**, in `kernels/flash_attention_backward.py` (built for
the FA backward-pass task just before this one, verified here on the same
GPU session): its backward kernel keeps more tiles resident per program
(K, V, plus each step's Q, dO) than the forward-only kernel (Q resident +
streamed K, V), so the 32x32 tile size proven safe for fp32/D=128 in
`kernels/README.md`'s kernel 4 v1 section wasn't safe for backward —
`OutOfResources: Required 107008, Hardware limit 101376` at fp32/D=128.
Fixed with a dtype/head-dim-aware tile size (`_pick_block_sizes`, 16x16 for
fp32 at D>64, 32x32 otherwise), same "verify on real hardware, don't
hand-estimate" habit as the forward kernel's own shared-memory config
probe.

Run for real on the L40S (inside the `vllm-openai` container, same as
`kernels/`):

```
$ python3 -m pytest kernels/tests/test_flash_attention_backward.py -v
...
23 passed in 32.52s   # includes all 4 test_gradcheck cases
$ python3 -m pytest model/tests/test_minimal_llama.py -v
...
14 passed in 42.26s
```

## CUDA Graph decode benchmark

`cuda_graph_decode.py` captures one decode step (embedding → n_layers ×
[attention + MLP] → final norm → lm_head, all at LLaMA-3-8B-Instruct's real
per-layer shape: hidden=4096, n_heads=32, head_dim=128,
intermediate=14336, vocab=128256 — `n_layers` truncated from the real 32,
see its docstring) as a single `torch.cuda.CUDAGraph`, then times the
decode loop both eager and graph-replayed, CUDA-event-timed, median of
post-warmup steps. Batch=1 — where per-kernel GPU work is smallest relative
to fixed per-launch CPU overhead, so where CUDA graphs matter most.

| n_layers | kernels/forward call | eager median (ms) | cuda_graph median (ms) | speedup |
|---|--:|--:|--:|--:|
| 4 (default) | 108 | 5.966 | 5.219 | 1.14x |
| 1 | 42 | 3.196 | 2.507 | 1.28x |

Kernel-launch counts are `torch.profiler`-measured (post-warmup, so they
reflect steady state, not the one-time `@triton.autotune` search cost —
see `_count_cuda_kernel_launches`'s docstring for a real bug this caught:
the first, un-warmed-up version of this measurement reported 7765 kernels
at n_layers=4, which was actually counting every matmul/attention
kernel's one-time autotune search as if every decode step paid it).

**The speedup is real but modest, and smaller at more layers, not
larger — worth explaining, not just reporting.** This file's decode step
reruns *full* `O(max_seq_len^2)` self-attention and every Linear at
`N=max_seq_len` (128) every step (see module docstring on why — no
incremental KV cache), so a real chunk of each step's time is genuine GEMM
compute (4096x14336 matmuls at N=128 are not free), not just kernel-launch
overhead — that compute grows with `n_layers` while the CPU-side launch
overhead CUDA graphs remove stays roughly proportional to kernel *count*,
so more layers dilutes the relative benefit (1.14x at 4 layers vs 1.28x at
1). A real incremental-KV-cache decode step (this repo's FA kernel doesn't
support the Nq != Nkv shape that needs, see `cuda_graph_decode.py`'s
docstring) would do genuinely single-token-sized compute per step, making
launch overhead a *larger* fraction of a much smaller total, and would be
expected to show a bigger relative speedup — flagged as real follow-up,
not done here.

Full data (per-step latencies, not just the summary above):
`cuda_graph_decode_results.json` (n_layers=1 run; n_layers=4's console
output is in this README's table above — same filename, only one run's
raw data is kept at a time).

Run for real on the L40S:

```
$ python3 -m model.cuda_graph_decode
Config: n_layers=4 hidden=4096 n_heads=32 head_dim=128 intermediate=14336 vocab=128256 max_seq_len=128 batch_size=1 dtype=torch.bfloat16
  108 CUDA kernels per forward call (n_layers=4) -> 1 graph.replay() call
mode          median (ms)    mean (ms)   p95 (ms)
eager               5.966        5.964      5.992
cuda_graph          5.219        5.246      5.480
Speedup (median eager / median cuda_graph): 1.14x

$ python3 -m model.cuda_graph_decode --n-layers 1
  42 CUDA kernels per forward call (n_layers=1) -> 1 graph.replay() call
mode          median (ms)    mean (ms)   p95 (ms)
eager               3.196        3.196      3.227
cuda_graph          2.507        2.564      2.662
Speedup (median eager / median cuda_graph): 1.28x
```

Note: run as `python3 -m model.cuda_graph_decode`, not
`python3 model/cuda_graph_decode.py` — the latter puts `model/` itself,
not the repo root, on `sys.path` and fails with
`ModuleNotFoundError: model` (this file's own absolute import of
`model.minimal_llama`).

## CI

`.github/workflows/kernels-ci.yml` now also runs `model/tests/` (renamed
scope in-place, still one workflow) — same "GitHub-hosted runners have no
GPU, so `requires_cuda` cases are skipped, this job catches import errors
and CPU-runnable validation only" story as `kernels/README.md`'s CI
section. `model/minimal_llama.py` imports straight from `kernels/`, so it
exercises the same CPU-only-triton-import path `kernels/tests` already
relies on.

## Files

- `minimal_llama.py` — the kernel-built forward pass (`llama_forward`) and
  its plain-PyTorch reference (`reference_llama_forward`), config/weight-init
  helpers.
- `tests/test_minimal_llama.py` — correctness tests (kernel-built vs
  reference, allclose).
- `cuda_graph_decode.py` — CUDA graph capture + eager-vs-graph decode-loop
  benchmark, CLI-configurable (`--n-layers`, `--max-seq-len`,
  `--batch-size`, `--num-steps`).
- `cuda_graph_decode_results.json` — raw per-step latencies from the
  `--n-layers 1` run above.
