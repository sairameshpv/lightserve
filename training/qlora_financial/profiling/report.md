# QLoRA training profiling: why v1 used about half the L40S

Profiling of the v1 QLoRA run (`../README.md`: 4h52m, ~1,340 tokens/s
training-only) with `../profile_train.py` on the same single L40S. Key findings:

- **The GPU starves inside forward/backward, not in data loading or the
  optimizer.** It is 51% busy. 99.8% of wall time is `fwd_bwd`, and the GPU is
  idle for 48.6% of it. Data loading and the optimizer step take <0.3% of each step.
- **Short sequences are the cause.** At batch 1, examples ≤256 tokens (43% of the
  data) train at 392 tokens/s versus a ~2,250 plateau: ~0.4 s of fixed overhead per
  micro-batch.
- **Fix: batch 4 + length grouping = 1,728 tokens/s (+29%)**, a projected full run
  of ~3h54m instead of 4h52m. Random batching of 4+ is *slower* (padding).
- **GPU utilization % is misleading here.** The slowest config (batch 8, random)
  shows 99% SM utilization: the GPU is busy computing padding.
- **4-bit dequantization is 20% of GPU kernel time**, the next target.
- **Measured follow-ups:** bf16 base + batch 4 + grouping ran at 1,816 tokens/s (+35%; GPU busy
  51% → 99%, dequantization gone), and keeping LoRA adapters in bf16 (PEFT upcasts them to fp32)
  reached **2,121 tokens/s (+58% over v1)**. Details at the end.

## Method

`profile_train.py` reuses `train.py`'s settings unchanged (4-bit base, LoRA r=16,
`max_length` 3,200), so "batch 1 × 16" is exactly v1. Each config trains 8 optimizer
steps on 128 real training examples, with a fixed seed. Tools:
- **PyTorch Profiler** (A): every kernel in steps 3-5, giving kernel categories and busy %.
- **Phase ranges** (NVTX + `record_function`) from Trainer hooks: `between_steps`
  (data loading + logging), `fwd_bwd`, `optimizer_step`, `post_step`. Gives GPU idle % per phase.
- **Nsight Systems**: same steps and phases, a separate run (the tools can't share CUPTI).
- **`nvidia-smi dmon`** (1 s samples) during B and C. Per-config SM util = mean over
  that config's training window (row timestamp in `sweep.log` minus its `seconds`).

```
python3 -m training.qlora_financial.profile_train --mode profile              # A
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi -o <out> \
  python3 -m training.qlora_financial.profile_train --mode profile --nsys     # A, nsys
python3 -m training.qlora_financial.profile_train --mode sweep                # B + C
```
(On the node: `sudo HF_HUB_OFFLINE=1 <venv>/bin/python`, as in `../README.md`.)

## A. Where a step's time goes: phases (v1 settings, steps 3-5)

| Phase | Wall per step | GPU idle | Share of wall time | nsys, wall per step |
|---|---|---|---|---|
| `between_steps` (data loading + logging) | 14.7 ms | 99.0% | 0.15% | 28.1 ms |
| `fwd_bwd` (16 micro-batches + grad clip) | 9,872 ms | **48.6%** | **99.78%** | 9,447 ms |
| `optimizer_step` | 2.2 ms | 61.0% | 0.02% | 2.5 ms |
| `post_step` (LR scheduler + zero_grad) | 4.9 ms | 73.4% | 0.05% | 5.7 ms |

Overall GPU busy: **51.3%** (kernel time / first-to-last-kernel span). Data loading and
the optimizer step are ruled out as causes of the idle time: together they are ~20 ms of a
~10 s step. The idle time is *inside* forward/backward. Gradient sync doesn't exist
on 1 GPU (no DDP; accumulation adds gradients in place). It is measured in the 2-GPU follow-up.

## A. What the GPU does when busy: kernels

GPU kernel time by category: matmul **56.5%**, 4-bit dequantize **19.7%**,
elementwise/norm 18.1%, memory copy 2.9%, attention 2.4%, loss 0.5%, optimizer ~0%,
other ~0%. That's 588,887 kernels in 3 steps, **~12,300 per micro-batch**.

| Top kernels (3 steps) | torch.profiler | nsys |
|---|---|---|
| `kDequantizeBlockwise` (bitsandbytes 4-bit → bf16) | 2,872 ms | 2,870 ms |
| cutlass bf16 gemm 256×128 (tn) | 2,561 ms | 2,563 ms |
| ampere bf16 gemm 256×128 (tn) | 1,488 ms | 1,492 ms |
| elementwise add (bf16) | 622 ms | 624 ms |
| cutlass bf16 gemm 128×128 (nn) | 619 ms | 619 ms |

The two tools agree within 0.3%. The single most expensive kernel is not a matmul but
**dequantization**, the price QLoRA pays for its 4-bit base.

## B. Tokens/s vs. sequence length (batch 1 × 16)

| Tokens per example | Share of train data | Time, 8 steps | Per micro-batch | Tokens/s | SM util |
|---|---|---|---|---|---|
| ≤256 | **42.9%** | 53.3 s | 0.42 s | **392** | 40% |
| 256-512 | 11.6% | 53.3 s | 0.42 s | 802 | 51% |
| 512-1,024 | 18.8% | 55.9 s | 0.44 s | 1,895 | 83% |
| 1,024-2,048 | 26.0% | 75.4 s | 0.59 s | 2,163 | 95% |
| 2,048-3,200 | 0.6% | 137.1 s | 1.07 s | 2,256 | 96% |

Up to ~1,000 tokens, a micro-batch costs a **fixed ~0.42 s regardless of length**: the
~12,300 small kernel launches, not the math, set the time. Only past ~1,000 tokens does
compute dominate, and throughput plateaus at ~2,250 tokens/s. So the 43% of examples ≤256
tokens run at ~17% of what the GPU achieves on long ones.

## C. Batch size × ordering (real length mix, effective batch 16)

| Batch × accum | Order | Tokens/s | vs. v1 | SM util | Peak mem | Final loss |
|---|---|---|---|---|---|---|
| 1 × 16 (v1) | random | 1,343 | n/a | 67% | 10.8 GiB | 2.528 |
| 2 × 8 | random | 1,379 | +3% | 94% | 10.8 GiB | 2.529 |
| 2 × 8 | length-grouped | 1,686 | +26% | 85% | 10.8 GiB | 2.887 |
| 4 × 4 | random | 1,071 | **−20%** | 97% | 9.4 GiB | 2.549 |
| **4 × 4** | **length-grouped** | **1,728** | **+29%** | 94% | 10.8 GiB | 2.891 |
| 8 × 2 | random | 893 | **−34%** | **99%** | 13.1 GiB | 2.549 |
| 8 × 2 | length-grouped | 1,519 | +13% | 100% | 12.9 GiB | 2.895 |

The batch-1 row reproduces v1: 1,343 vs. v1's ~1,340 tokens/s training-only (4h52m minus
~30 min of eval passes). Tokens/s counts real tokens only, not padding. Random-order losses
should match across batch sizes (same examples, same order); grouped ones differ because
grouping changes the order.

## Reading this

1. **Batch 1 is launch-bound.** Each micro-batch issues ~12,300 small kernels, one set per
   layer for base matmuls, dequantize, 7 LoRA adapters, norms and activations, plus the
   gradient-checkpointing recompute. For short sequences each kernel finishes before the CPU
   has queued the next, so the GPU waits (A: 48.6% idle inside `fwd_bwd`; B: flat 0.42 s).
2. **Batching fixes that only if lengths match.** A batch pads to its longest member. With
   random order, a batch of 4 or 8 is usually padded to a long example, so the GPU does
   *more* total work: the −20% / −34% rows. Grouping by length keeps padding small, so
   batching's launch savings survive (+26% / +29%).
3. **SM utilization % is not useful work.** `nvidia-smi` reports the share of time *any*
   kernel runs. Random batch 8 is the slowest config at the highest utilization (99%):
   it's busy multiplying padding. Throughput in real tokens/s is the metric to trust.
4. **Why batch 8 grouped trails batch 4 grouped** (1,519 vs 1,728) was not investigated.
   Likely candidates: once launch overhead is amortized, larger batches only add padding
   within imperfect groups, and long-example batches (up to ~25k tokens) grow activation
   and recompute cost. A hypothesis, not a measured cause.

## Recommendation

Train with **`per_device_train_batch_size=4`, `gradient_accumulation_steps=4`,
`train_sampling_strategy="group_by_length"`**. It keeps the same effective batch of 16 and
fits in memory with room to spare (10.8 of 46 GB). Projected full run: 21.09M tokens ÷
1,728 tokens/s = 3h23m of training, plus the unchanged ~30 min of eval, ≈ **3h54m vs.
4h52m (−20%)**. That's a projection from 128-example runs, not a measured full run (see caveats).

Next targets, from what A measured (each still needs its own measurement):
- **Dequantization (20% of kernel time).** With ~35 GB unused, plain LoRA on a bf16 base
  (no 4-bit) may fit. It would remove this cost, but it changes the method from QLoRA to LoRA.
- **Launch overhead.** Fewer, fused kernels: TRL's `use_liger_kernel`, or `torch.compile`.

## Caveats and methodology notes

- **Small samples.** Each config ran 8 steps on 128 examples, one run each, first-step
  warm-up included. `group_by_length` over the full 34k set may group better or worse.
- **Quality not measured.** Grouping changes batch composition; its effect on the fine-tune's
  accuracy is unmeasured. Random batches ≥4 shifted the 8-step loss by 0.8% (2.549 vs.
  2.528), which is small, consistent, and not investigated.
- **Profilers slow training**: 809 tokens/s under torch.profiler, 966 under nsys, 1,343
  without. Absolute speeds come only from unprofiled runs (B, C).
- **Phase timing differs by tool** for the short phases (`between_steps` 14.7 vs. 28.1
  ms/step). nsys captured 2 of the 3 data-loading windows fully, and the tools time CPU
  ranges differently. Both put it below 0.3% of a step.
- **The first A run was wrong and was rerun** (fixed in 1df81c6): phase ranges counted as
  kernels (busy 144.6%), the profiler schedule repeating (1 step exported, not 3), and
  dropout uncategorized.
- **Session interruptions**: the preemptible node was reclaimed mid-rerun (A rerun from
  scratch) and hung once (SSH dead while `RUNNING`; stop/start fixed it). No result mixes runs.

## Applying the recommendations: v2_lora (measured)

Instead of a full retrain, one ~20-min **profiled** run applied both recommendations:
bf16 base (plain LoRA, no 4-bit) + batch 4 × 4 + `group_by_length`, 150 steps on 2,400
random training examples (vs. 128 before), 13m40s. The profiler recorded steps 3-5; the other
147 ran unprofiled.
```
profile_train --mode profile --name v2_lora --bf16-base --batch-size 4 --grad-accum 4 --group-by-length --steps 150
```

| Config | Tokens/s | vs. v1 |
|---|---|---|
| v1: QLoRA 4-bit, batch 1, random (C sweep) | 1,343 | n/a |
| QLoRA 4-bit, batch 4, grouped (C sweep) | 1,728 | +29% |
| **v2_lora: bf16, batch 4, grouped** | **1,816** | **+35%** |

v2_lora's figure is the whole run including the 3 profiled steps, so slightly understated.
The +5% over QLoRA batch 4 is approximate: different sample (2,400 vs. 128 examples) and session.

| Profile, steps 3-5 | v1 | v2_lora |
|---|---|---|
| GPU busy | 51.3% | **98.8%** |
| GPU idle inside `fwd_bwd` | 48.6% | **1.1%** |
| Kernels launched (3 steps) | 588,887 | 148,015 |
| Kernels per micro-batch | ~12,268 | ~12,335 |
| **Kernels per example** | ~12,268 | **~3,084 (~4× fewer)** |
| 4-bit dequantize share of GPU time | 19.7% | **0% (gone)** |
| Data loading (`between_steps`) / optimizer, per step | 14.7 / 2.2 ms | 13.1 / 2.3 ms |
| Peak GPU memory | 6.6 GiB | 23.0 GiB (of 46) |

**Why it's faster:**
1. **Batching amortizes the launches.** The model still issues the same ~12,300 kernels per
   micro-batch, but each now carries 4 examples, so ~4× fewer launches per example. The CPU
   keeps up, and idle inside `fwd_bwd` falls from 48.6% to 1.1%. This directly confirms the
   launch-bound explanation in *Reading this*, which was inferred before.
2. **Removing dequantization helps less than its 19.7% suggested** (~+5% over QLoRA at batch 4).
   Dequantizing a layer's weights costs the same whether the micro-batch holds 1 example or 4,
   so at batch 4 that cost was already spread over 4 examples. It matters most at batch 1.

**The new bottleneck.** With the GPU 99% busy, the question becomes what it's busy with: matmul
39.2%, **elementwise/norm 34.0%, memory copy 14.1%**, attention 12.5%. Nearly half is not matmul.
Top kernels include `bfloat16_copy_kernel` (2,388 ms) and `fused_dropout_kernel_vec<float, ...>`
(2,407 ms, fp32), which suggests PEFT keeps the LoRA adapters in fp32 and converts activations
around them. **Confirmed** by the next experiment: see *bf16 adapters (measured)*.

**Caveats**: steps 3-5 are among the longest batches (`group_by_length` puts the longest batch
first), which inflates attention's share; busy % and dequantize-gone don't depend on this.
Accuracy is unmeasured (loss fell normally, 0.596 at step 150). Projected full run at 1,816 tok/s:
3h14m of training + ~30 min eval ≈ **3h44m vs. 4h52m (−23%)**, a projection (eval time in bf16
not measured).

## bf16 adapters (measured)

**Cause, from the sources and then measured.** PEFT 0.21's `get_peft_model` defaults to
`autocast_adapter_dtype=True`, which upcasts bf16 adapter weights to fp32. TRL 1.14's
`SFTTrainer` casts trainable params back to bf16 **only for quantized models**. So v1 (4-bit)
trained bf16 adapters, while v2_lora (bf16 base) trained fp32 ones: the run recorded
`adapter_dtype_before = torch.float32`. **Fix**: `--bf16-adapters` applies TRL's own QLoRA-path
cast before `train()` (now `torch.bfloat16`). One change versus v2_lora, same command otherwise.

| | v2_lora (fp32 adapters) | **+ `--bf16-adapters`** |
|---|---|---|
| Tokens/s (whole run, 150 steps) | 1,816 | **2,121 (+17%)** |
| Kernels, steps 3-5 (per micro-batch) | 148,015 (~12,335) | **115,759 (~9,647), −22%** |
| Memory copy / elementwise / matmul | 14.1% / 34.0% / 39.2% | **4.5%** / 33.1% / 46.9% |
| Dropout kernel | `<float>`, 2,407 ms | `<BFloat16>`, 846 ms |
| `bfloat16_copy` + `direct_copy` kernels | 2,388 + 2,138 ms | not in the top 10 |
| GPU busy / peak memory / loss at step 150 | 98.8% / 23.0 GiB / 0.596 | 98.7% / 21.5 GiB / 0.568 |

**Reading.** The GPU was already ~99% busy, so the +17% is not more utilization but **less
unnecessary work**: no fp32 casts around each adapter, so fewer kernels and more of the time in
matmuls. Adapter grads and optimizer state in bf16 also lower peak memory.

| The whole chain (rows 1-2: 128-example C sweep; rows 3-4: 2,400-example runs) | Tokens/s | vs. v1 |
|---|---|---|
| v1: QLoRA 4-bit, batch 1, random | 1,343 | n/a |
| + batch 4 × 4, `group_by_length` (4-bit) | 1,728 | +29% |
| + bf16 base (v2_lora) | 1,816 | +35% |
| **+ bf16 adapters** | **2,121** | **+58%** |

Projected full run at 2,121 tok/s: 2h46m of training + ~30 min eval ≈ **3h16m vs. 4h52m
(−33%)**, a projection. Accuracy is unmeasured; v1 trained bf16 adapters to 74.5%, and loss stayed
stable here. Next candidate: elementwise/norm, still 33% (fused kernels: `use_liger_kernel`,
`torch.compile`).

## Files

Committed (small, in this folder):
- `profile_v1.json`: experiment A (kernel categories, top kernels, busy %, phases).
- `profile_v1_nsys_run.json`: tokens/s of the nsys run of A.
- `nsys_nvtx_sum.csv`, `nsys_cuda_gpu_kern_sum.csv`: `nsys stats` summaries of that run.
- `results.json` / `results.csv`: experiments B and C, one row per config.
- `profile_v2_lora.json`: the v2_lora follow-up run (whole-run tokens/s + steps 3-5 profile).
- `profile_v2_lora_bf16ad.json`: the same plus `--bf16-adapters`, incl. adapter dtype before/after.

Local only (not committed, see `.gitignore`): `logs/` (run logs + `nvidia-smi dmon`
samples used for SM util), `traces/profile_v1_nsys.nsys-rep` (32 MB, open in
`nsys-ui`). The 1.58 GB torch.profiler Chrome trace exists only on the node's disk.
