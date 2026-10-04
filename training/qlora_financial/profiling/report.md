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
  51% → 99%, dequantization gone), keeping LoRA adapters in bf16 (PEFT upcasts them to fp32)
  reached 2,121 tokens/s (+58%), and Liger fused kernels **2,239 tokens/s (+67% over v1)**.
  Details at the end.
- **Confirmed end to end:** a full v2 retrain with all four changes took **3h00m vs. 4h52m
  (−38%) at the same accuracy** (75.0% vs. 74.5%), as projected.
- **DDP on 2 GPUs: 1.86× (92.9% scaling efficiency).** Gradient sync costs ~57 ms of network per
  update; most of the remaining gap is one GPU waiting for the other (load imbalance), not the network.
- **Partial gradient checkpointing: 2,460 tokens/s (+9.9%, +83% over v1)** by skipping it on 8 of 32
  layers, at 39.7 GiB peak memory. A flaw in my short memory probes (too-short batches) is written up.
- **FlashAttention-2 + padding-free batches: 2,579 tokens/s (+4.8%, +92% over v1).** Padding forced
  PyTorch's slower attention kernel; attention fell from 18.3% to 4.0% of GPU time, memory to 33.1 GiB.

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

## Liger fused kernels (measured)

One change versus the bf16-adapter run: `--liger` (TRL's `use_liger_kernel=True`, with
`pip install liger-kernel` 0.8.4; torch stayed 2.11.0+cu128). Liger replaces Llama's RMSNorm,
RoPE, SwiGLU and cross-entropy with fused Triton kernels. Same 150 steps and examples.

| | bf16 adapters | **+ `--liger`** |
|---|---|---|
| Tokens/s (whole run) | 2,121 | **2,239 (+6%)** |
| Time for 150 steps | 701.8 s | 664.8 s |
| Kernels, steps 3-5 | 115,759 | **81,139 (−30%)** |
| Elementwise/norm / matmul / attention | 33.1% / 46.9% / 15.4% | **24.2%** / 57.1% / 18.0% |
| Memory copy | 4.5% | 0.6% |
| GPU busy / peak memory / loss at step 150 | 98.7% / 21.5 GiB / 0.568 | 98.5% / 20.9 GiB / 0.560 |

**Reading: diminishing returns.** Fusion cut kernels by 30% but run time only ~5%. The removed
kernels were small elementwise passes, while the big costs (matmul 57%, attention 18%) are
untouched by Liger and now dominate. Loss stayed consistent (0.560 vs. 0.568), so the fused kernels
compute the same thing. The chain from the bf16-adapter section extends to **+ Liger: 2,239
tokens/s, +67% over v1**. Projected full run: 2h37m of training + ~30 min eval ≈ **3h07m vs.
4h52m (−36%)**, a projection with accuracy unmeasured.

Next candidates, both unmeasured: **turning off gradient checkpointing** (it recomputes the forward
pass during backward, and ~21 of 46 GB is used, so memory may now allow it), and **attention**
(18%, the memory-efficient SDPA kernel; FlashAttention-2 isn't installed).

**Two bugs in `profile_train.py`, found in this run (fixed in 5ad417b):**
- **The first attempt crashed before training**: `run_config()` copied `loss_type="chunked_nll"`
  (resolved by TRL while Liger was off), which TRL rejects with Liger. The rerun used
  `loss_type="nll"`, TRL's own choice with Liger.
- **`liger_applied: false` in `profile_v2_lora_bf16ad_liger.json` is wrong.** transformers 5.17
  applies Liger *inside* `Trainer.train()`, and the check ran before it. Liger did run: the
  run's trace contains `_rms_norm_forward_kernel` ×1,548, `_triton_rope` ×1,152,
  `_swiglu_forward_kernel` ×768 and `liger_cross_entropy_kernel` ×261, versus **0** of each in
  the bf16-adapter trace (counted with `grep` on the node). The check now runs after `train()`,
  and the profile also records `liger_kernel_launches` as direct evidence.

## Full v2 run: projection confirmed

A full retrain with all four changes (`../README.md`, *v2*) measured what the 150-step runs
projected:

| | Projected (from 150-step runs) | Measured (full run) |
|---|---|---|
| Training-only throughput | 2,239 tok/s | **2,258 tok/s** |
| Training-only time | 2h37m | 2h36m (9,340.1 s) |
| Total incl. 9 eval passes | ~3h07m (v1's eval time assumed) | **3h00m** (eval passes faster: 1,485 vs. 1,816 s) |
| vs. v1 (4h52m) | −36% | **−38%** |

Accuracy on the same 200 test questions: **75.0% vs. v1's 74.5%**, a tie (11 vs. 12 questions
right only in one run). The open caveat throughout this report, "accuracy unmeasured", is now
answered: the speedups cost no accuracy. Numbers: `v2_full_run.json`.

## DDP on 2 GPUs (measured)

The same profiled run (v2 settings: bf16 base + adapters, Liger, `group_by_length`) on **2
L40S**, one per node (the Nebius preset has 1 GPU per node), with PyTorch DDP: `torchrun --nnodes 2
--nproc_per_node 1`, NCCL 2.28.9 over the nodes' internal network (TCP, no InfiniBand).
Batch 4 × 2 accumulation × 2 GPUs = **16 per update**, the same 2,400 examples and 150 updates as the
1-GPU run, so the two are directly comparable.

| | 1 GPU | DDP rank 0 | DDP rank 1 |
|---|---|---|---|
| Time for 150 updates | 664.8 s | **357.7 s** | 357.7 s |
| Tokens/s (global) | 2,239 | **4,162** | 4,162 |
| **Speedup / scaling efficiency** | n/a | **1.86× / 92.9%** | |
| GPU busy / idle inside `fwd_bwd` | 98.5% / 1.3% | 98.2% / 1.5% | 98.2% / 1.6% |
| Gradient sync (NCCL) share of GPU time | n/a | 1.5% | **6.1%** |
| Peak memory | 20.9 GiB | 21.0 GiB | 19.4 GiB |

**Where the missing 7% goes: waiting, not the network.** A one-off script timed a raw cross-node
all-reduce of 84 MB (≈ the LoRA gradients synced per update): median **57.5 ms** over 20 runs (min
57.1, max 75.7; ~1.46 GB/s). In training, rank 1's `ncclDevKernel_AllReduce` took 533 ms over the 3
profiled steps, **~178 ms per update**. NCCL's kernel runs until both GPUs arrive, so the extra
**~120 ms is rank 1 waiting for rank 0**, which had more work that step. Rank 0 rarely waits (1.5%).
The imbalance is largest on the longest batches: in a 5-step smoke run, all from the start of
`group_by_length`'s longest-first order, rank 1 spent 36.2% of its GPU time in NCCL. Grouping evens
out lengths within a batch, not the work given to each GPU. Unmeasured next idea:
`train_sampling_strategy="batch_rebalance"`, which balances padded-token cost across devices.

**Training is equivalent.** Mean loss over the 150 per-step logs: DDP 0.966 vs. 1 GPU 0.988 (per
50-step segment 1.554/1.588, 0.752/0.714, 0.591/0.661). The single step-150 loss differs (0.472 vs.
0.560) because grouping runs per process (4 × 2 = 8 examples) rather than per update (16), which
changes which examples share a batch. Same-loss check: both ranks report identical losses (synced).
DDP config: `ddp_find_unused_parameters=False`. transformers 5.17 auto-disables it only for a plain
`PreTrainedModel`, and ours is a `PeftModel`, where the default `True` clashes with gradient checkpointing.

**Three profiling bugs from the first DDP smoke run (fixed in b66d800):**
- *Clock skew*: rank 1 finished loading first and timed its wait for rank 0 (157.6 s vs. 30.6 s for
  the same 5 steps). Fix: barrier before starting the clock (both 28.1 s after).
- *Labels counted as kernels*: DDP and NCCL add GPU-side labels (`DistributedDataParallel.forward`,
  `nccl:...`), giving busy 129.7% / 144.2%. Fix: exclude `is_user_annotation` events.
- *Overlapping streams*: NCCL runs alongside compute, so summed kernel time exceeds the span. Fix:
  busy time = union of kernel intervals.

## Partial gradient checkpointing (measured)

One change versus the Liger run (1 GPU): `--ckpt-skip-layers 8`. Gradient checkpointing saves
memory by discarding each layer's intermediate results in the forward pass and recomputing them in
backward. transformers 5.17 gives every decoder layer its own on/off flag, so a callback switches it
off on the first 8 of 32 layers at the start of training; those 8 keep their activations instead of
recomputing them. Same 150 steps and 2,400 examples.

| | All 32 layers checkpointed | **First 8 not checkpointed** |
|---|---|---|
| Tokens/s (whole run) | 2,239 | **2,460 (+9.9%)** |
| Time for 150 steps | 664.8 s | **605.2 s (−9.0%)** |
| Kernels, steps 3-5 | 81,139 | 75,992 (−6.3%) |
| GPU busy | 98.5% | 98.5% |
| Peak memory | 20.9 GiB | **39.7 GiB** (of 44.4 GiB total) |
| Loss at step 150 | 0.560 | 0.545 |

**Why +9.9%, more than the ~6% I first estimated.** My estimate assumed the usual rule that backward
costs ~2× forward, which makes recomputation ~1/4 of the work. Here the base weights are frozen, so
backward computes only the gradients flowing back through each layer (not weight gradients for the
base), and backward costs only about 1× forward. Recomputation is then ~1/3 of the work: 8 of 32
layers × 1/3 ≈ 8.3% less time, ≈ +9% tokens/s, close to the measured −9.0% / +9.9%. By the same
rule, turning checkpointing off on all 32 layers would be worth ~+50%, but at ~2.34 GiB per layer
(next paragraph) it would need ~96 GiB, over twice the GPU. GPU busy is unchanged (98.5%): the GPU
was already fully used, and the gain comes from doing less work, not from less waiting. The loss
difference (0.545 vs. 0.560) is the same size as the Liger run's (0.560 vs. 0.568): small numeric
differences in bf16, not a change in what is learned.

**How 8 was chosen, and a flaw in my short test runs.** Checkpointing fully off ran out of memory
(OOM) at 44.35 GiB, as expected. I then ran 5-step probes to measure memory per layer: K=28 layers
off ran OOM (44.38 GiB), K=4 peaked at 23.7 GiB and K=16 at 39.8 GiB, so (39.8 − 23.7) / 12 = 1.34
GiB per layer. But the real 150-step run with K=16 then ran OOM at step 1 (44.32 GiB). The cause:
profile mode draws steps × 16 examples, so a 5-step probe sees only 80 examples, and with
longest-first grouping its first batch is 4 × 1,635 = 6,540 tokens. The 150-step run's 2,400 examples
start with 4 × 2,854 = 11,416 tokens, 1.75× longer, and activation memory grows with tokens. Rescaled:
1.34 × 1.75 ≈ 2.34 GiB per layer, so K=8 predicts 20.9 + 8 × 2.34 = 39.6 GiB; measured: **39.7**.
Lesson: a memory probe must run the real worst-case batch, not just fewer steps. A fix (not
implemented): let the probe's step count differ from its sample size. The probe JSONs are committed
as evidence; the three OOMs are in the gitignored run logs.

**The 1-GPU chain extends to 2,460 tokens/s, +83% over v1** (1,343 → 1,728 → 1,816 → 2,121 → 2,239
→ 2,460). Projected full run: the v2 run's training tokens at 2,460 tokens/s take 8,573 s (2h23m),
plus v2's measured eval time (1,485 s) ≈ **2h48m vs. v2's 3h00m (−7%) and v1's 4h52m (−43%)**, a
projection, not measured. **Caution before a full run with K=8:** the full dataset's longest batch is
4 × 3,196 = 12,784 tokens, 1.12× the longest profiled one. Scaling memory per layer the same way
gives ~41.9 GiB, under 3 GiB from the 44.4 GiB limit, which is tight given the estimate is rough.
**K=7 (~39.2 GiB) is the safer setting for a full run**, giving up about 1/8 of the gain.

## FlashAttention-2 + padding-free batches (measured)

**Why attention used the slower kernel.** After partial checkpointing, attention was 18.3% of GPU
time, all in PyTorch's *memory-efficient* kernels (`PyTorchMemEffAttention`), not its faster
*flash* ones. The cause is padding: a batch of 4 almost always pads its shorter examples, so
transformers 5.17 builds a mask marking the padding (`masking_utils.py`, which skips the mask only
when nothing is padded). PyTorch's flash kernel can't take a mask, so SDPA falls back to
memory-efficient (`integrations/sdpa_attention.py`).

**The change:** `--flash-attn` (commits 5eb2b4e, 977ef1f) turns on two things together.
**Padding-free batches** (TRL `padding_free=True`): the batch's examples are placed end to end in one
row with no padding, and position numbers restart at each example. **FlashAttention-2's "varlen"
kernel** uses those boundaries to keep examples separate, so no mask is needed. The kernel comes
prebuilt from the Hugging Face Hub (`kernels-community/flash-attn2@v2`, via `pip install
"kernels>=0.16,<0.17"`), so nothing is compiled. Everything else matches the K=8 run.

| | SDPA (memory-efficient) | **FlashAttention-2 + padding-free** |
|---|---|---|
| Tokens/s (whole run) | 2,460 | **2,579 (+4.8%)** |
| Time for 150 steps | 605.2 s | **577.2 s (−4.6%)** |
| Time per profiled step (steps 3-5) | 8,237 ms | 7,325 ms (−11.1%) |
| Attention / matmul / elementwise+norm | 18.3% / 56.9% / 24.2% | **4.0%** / 67.8% / 28.0% |
| Peak memory | 39.7 GiB | **33.1 GiB (−6.6)** |
| GPU busy | 98.5% | 95.6% |
| Kernels, steps 3-5 | 75,992 | 87,204 (+14.8%) |
| Mean loss over the 150 steps | 0.982 | 0.980 |

**Reading: attention is mostly solved; the whole-run gain is smaller than the per-step one.**
Attention fell from 18.3% to 4.0% of GPU time, and matmul (68%) dominates again. The profiled steps
gained 11%, but the whole run only 4.6%. A likely reason, not checked: with longest-first ordering,
steps 3-5 are among the longest batches, and attention's cost grows faster than length, so it saves
the most there and less on the shorter batches later. Memory fell 6.6 GiB, room for more layers
without checkpointing (unmeasured). **Two open questions:** kernels went *up* 15% and GPU busy
*down* (98.5% → 95.6%); possibly the padding-free path adds small kernels and some launch waiting
returns, not investigated. The 1-GPU chain extends to **2,579 tokens/s, +92% over v1**. Projected full
run: 8,178 s of training + 1,485 s eval ≈ **2h41m vs. v2's 3h00m (−11%) and v1's 4h52m (−45%)**, a
projection, not measured.

**Training is unchanged.** A 5-step check on the same 80 examples as the two SDPA memory probes gave
step-5 loss 3.2315 vs. 3.2289 / 3.2287 (`profile_flash_smoke.json`). Over 150 steps, mean loss per
50-step block: 1.581 / 0.704 / 0.654 vs. SDPA 1.573 / 0.719 / 0.652; median per-step difference
0.014 (from the gitignored run logs). The step-150 loss alone differs more (0.618 vs. 0.545), but
single steps swing between 0.18 and 0.68 at that point.

**Three problems found on the node, all fixed:**
- *Wrong `kernels` version:* pip installed 0.17.2, and transformers 5.17 requires 0.16.x.
- *TRL refuses `padding_free` with a `max_length`* (it can't cut examples in that mode). Fix:
  `max_length=None` with `--flash-attn`. Nothing changes: the longest example is 3,197 tokens.
- *The default kernel build crashed in the backward pass.* transformers asks for "version 3", a single
  "stable ABI" build meant to work across PyTorch versions; on torch 2.11 its backward called
  `aten::sum` with a garbage dimension. Fix: pin `@v2`, which has a build for torch 2.11 + CUDA 12.8,
  after checking its forward output and gradients against PyTorch's own attention (bf16-rounding level).

The Hub kernel's version lookup needs network access, so these runs used the HF token instead of
`HF_HUB_OFFLINE=1` (the model itself still loaded from the local cache).

## Files

Committed (small, in this folder):
- `profile_v1.json`: experiment A (kernel categories, top kernels, busy %, phases).
- `profile_v1_nsys_run.json`: tokens/s of the nsys run of A.
- `nsys_nvtx_sum.csv`, `nsys_cuda_gpu_kern_sum.csv`: `nsys stats` summaries of that run.
- `results.json` / `results.csv`: experiments B and C, one row per config.
- `profile_v2_lora.json`: the v2_lora follow-up run (whole-run tokens/s + steps 3-5 profile).
- `profile_v2_lora_bf16ad.json`: the same plus `--bf16-adapters`, incl. adapter dtype before/after.
- `profile_v2_lora_bf16ad_liger.json`: the same plus `--liger` (its `liger_applied: false` is a
  bug; see *Liger fused kernels*).
- `profile_ddp2_rank0.json` / `profile_ddp2_rank1.json`: the 2-GPU DDP run, one file per GPU.
- `profile_ckpt_skip8.json`: partial gradient checkpointing (first 8 layers not checkpointed).
- `profile_ckpt_skip4_probe.json` / `profile_ckpt_skip16_probe.json`: the 5-step memory probes,
  kept as evidence of the probe flaw (their batches were shorter than the real run's).
- `profile_ckpt_skip8_flash.json`: the same plus `--flash-attn` (records `attn_implementation`).
- `profile_flash_smoke.json`: its 5-step check (same examples as the probes; loss matches).
- `v2_full_run.json`: the full v2 vs. v1 retrain (runtimes, eval curves, accuracy, per-question
  overlap), built from the gitignored `outputs/mlflow.db` and both `verify_results.jsonl`.

Local only (not committed, see `.gitignore`): `logs/` (run logs + `nvidia-smi dmon`
samples used for SM util), `traces/profile_v1_nsys.nsys-rep` (32 MB, open in
`nsys-ui`). The 1.58 GB torch.profiler Chrome trace exists only on the node's disk.
