"""Profile QLoRA training: why the v1 run used ~37% of the L40S, and what fixes it.

v1 (train.py, batch 1 x 16 accumulation) ran at ~1,200 tokens/s overall but
~2,100 on the smoke run's longest examples, with peak memory 7.4 of 46 GB --
pointing at short sequences at batch 1 leaving the GPU idle. Three experiments:
  A (--mode profile): torch.profiler over a few v1-setting steps -> GPU time
    by kernel category, GPU busy % (kernel time / wall time), top kernels, and
    GPU idle % per loop phase (between_steps = data loading + logging, fwd_bwd,
    optimizer_step, post_step) from NVTX/record_function ranges. --nsys: the
    same phases under Nsight Systems as a cross-check. No gradient sync to
    measure on 1 GPU (no DDP) -- that's for the 2-GPU follow-up.
  B (--mode sweep):   tokens/s vs sequence length at batch 1.
  C (--mode sweep):   batch 1/2/4/8 (effective batch kept at 16), random vs
    length-grouped order -> tokens/s, peak memory, loss (should match).
Reuses train.py's load_split / qlora_configs / sft_config unchanged. GPU only.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["profile", "sweep"], required=True)
    ap.add_argument("--steps", type=int, default=8, help="optimizer steps per measured config")
    ap.add_argument("--out-dir", default="training/qlora_financial/profiling")
    ap.add_argument("--data-dir", default="training/qlora_financial/data")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--nsys", action="store_true", help="profile mode under `nsys --capture-range=cudaProfilerApi`: "
                    "no torch.profiler (they can't share CUPTI); NVTX phases + capture of steps 3-5 only")
    return ap.parse_args()


# Token-length buckets for experiment B (upper bound 3200 = train.py's max_length).
BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 3200)]
# train_sampling_strategy value for length-grouped batches -- confirmed in transformers
# 5.17's training_args.py choices: random, sequential, group_by_length, batch_rebalance.
GROUPED = "group_by_length"


def token_lengths(ds, model_name: str) -> list:
    """Tokens in each example's full conversation (prompt + answer) -- the same
    count prepare_dataset.py's length audit and train.py's max_length use."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    texts = [tok.apply_chat_template(p + c, tokenize=False) for p, c in zip(ds["prompt"], ds["completion"])]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def bucket_by_length(ds, lengths: list, per_bucket: int, seed: int = 0) -> dict:
    """{"256-512": Dataset, ...}: per_bucket random examples from each BUCKET
    (fixed seed, so every run measures the same examples)."""
    rng = random.Random(seed)
    out = {}
    for lo, hi in BUCKETS:
        idx = [i for i, n in enumerate(lengths) if lo < n <= hi]
        out[f"{lo}-{hi}"] = ds.select(sorted(rng.sample(idx, min(per_bucket, len(idx)))))
    return out


def run_config(args, train_ds, batch_size: int, grad_accum: int, sampling: str = "random", callbacks=()):
    """Train `args.steps` optimizer steps with train.py's exact settings, changing
    only batch size / accumulation / example order; eval, saving, MLflow off."""
    import dataclasses
    from types import SimpleNamespace
    from training.qlora_financial.train import sft_config
    base = sft_config(SimpleNamespace(output_dir=str(Path(args.out_dir) / "tmp"), smoke=False, max_length=3200))
    config = dataclasses.replace(
        base, per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        train_sampling_strategy=sampling, max_steps=args.steps, logging_steps=1,
        eval_strategy="no", save_strategy="no", report_to="none")
    import time
    import torch
    from trl import SFTTrainer
    from training.qlora_financial.train import qlora_configs
    quant, lora = qlora_configs()
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=train_ds,
                         quantization_config=quant, peft_config=lora, callbacks=list(callbacks))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()  # timed without model loading; includes the first step's warm-up
    secs = time.perf_counter() - t0
    last = [h for h in trainer.state.log_history if "num_tokens" in h][-1]  # real (non-pad) tokens, cumulative
    return {"batch_size": batch_size, "grad_accum": grad_accum, "sampling": sampling, "seconds": round(secs, 1),
            "tokens_per_s": round(last["num_tokens"] / secs), "final_loss": last["loss"],
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}


# GPU kernel name -> category, first match wins. Order matters: attention/loss come
# before matmul because flash kernels' template args mention "cutlass". Substrings
# are a first guess from typical CUDA/bitsandbytes names; summarize_profile() reports
# whatever lands in "other" so the list can be corrected against the real names.
KERNEL_CATEGORIES = [
    ("4-bit dequantize", ["dequantize", "kdequant"]),
    ("attention", ["flash", "fmha", "attention", "attn"]),
    ("loss", ["cross_entropy", "nll_loss", "log_softmax", "logsoftmax"]),
    ("matmul", ["gemm", "gemv", "cutlass", "cublas", "xmma", "nvjet", "ampere_", "sm80_", "sm89_"]),
    ("optimizer", ["adam", "multi_tensor", "foreach"]),
    ("memory copy", ["memcpy", "memset", "copy"]),
    ("elementwise/norm", ["elementwise", "vectorized", "reduce", "norm", "silu", "softmax", "index", "cat", "dropout"]),
]


def categorize(kernel_name: str) -> str:
    name = kernel_name.lower()
    return next((cat for cat, subs in KERNEL_CATEGORIES if any(s in name for s in subs)), "other")


def make_profiler_callback(trace_path: str, wait: int = 1, warmup: int = 1, active: int = 3):
    """Trainer hook running torch.profiler per *optimizer* step: skip `wait` steps
    (first-step setup), warm up, then record `active` steps and save a Chrome trace."""
    import torch
    from transformers import TrainerCallback
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    class ProfilerCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.prof = torch.profiler.profile(activities=acts, schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1))  # default repeats: export = last cycle only
            self.prof.__enter__()

        def on_step_end(self, args, state, control, **kw):
            self.prof.step()

        def on_train_end(self, args, state, control, **kw):
            self.prof.__exit__(None, None, None)
            self.prof.export_chrome_trace(trace_path)
    return ProfilerCallback()


def make_phase_callback(nsys_capture: bool = False):
    """Marks training-loop phases as NVTX ranges (nsys) + record_function ranges
    (torch.profiler). Boundaries follow transformers 5.17's loop: all 16 micro-batches
    are fetched *before* on_step_begin, so data loading lands in between_steps."""
    import torch
    from transformers import TrainerCallback

    class PhaseCallback(TrainerCallback):
        current = None  # the open record_function range, if any

        def _switch(self, name):
            if self.current is not None:
                self.current.__exit__(None, None, None)
                torch.cuda.nvtx.range_pop()
            self.current = None
            if name:
                torch.cuda.nvtx.range_push(name)
                self.current = torch.profiler.record_function(name)
                self.current.__enter__()

        def on_train_begin(self, *a, **kw): self._switch("between_steps")  # data fetch + logging
        def on_step_begin(self, args, state, control, **kw):
            if nsys_capture and state.global_step == 2:  # same steps 3-5 torch.profiler records
                torch.cuda.profiler.start()
            self._switch("fwd_bwd")  # 16 micro-batches + grad clip
        def on_pre_optimizer_step(self, *a, **kw): self._switch("optimizer_step")
        def on_optimizer_step(self, *a, **kw): self._switch("post_step")  # lr scheduler + zero_grad
        def on_step_end(self, args, state, control, **kw):
            self._switch("between_steps")
            if nsys_capture and state.global_step == 5:
                torch.cuda.profiler.stop()
        def on_train_end(self, *a, **kw): self._switch(None)
    return PhaseCallback()


PHASES = ("between_steps", "fwd_bwd", "optimizer_step", "post_step")


def phase_breakdown(prof) -> dict:
    """Per phase: wall time (the CPU-side range), GPU kernel time *executing inside
    that wall window* (intervals clipped to it), and GPU idle % -- the starvation
    number. Attribution is by when kernels ran, not by which phase launched them."""
    import torch
    events = prof.events()
    kernels = [(e.time_range.start, e.time_range.end) for e in events  # phase names also appear as GPU annotations
               if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in PHASES]
    out = {}
    for name in PHASES:
        wins = [(e.time_range.start, e.time_range.end) for e in events
                if e.name == name and e.device_type == torch.autograd.DeviceType.CPU]
        wall = sum(b - a for a, b in wins)
        gpu = sum(max(0, min(b, kb) - max(a, ka)) for a, b in wins for ka, kb in kernels)
        out[name] = {"count": len(wins), "wall_ms": round(wall / 1000, 1), "gpu_ms": round(gpu / 1000, 1),
                     "gpu_idle_pct": round(100 * (1 - gpu / wall), 1) if wall else None}
    return out


def summarize_profile(prof) -> dict:
    """GPU busy % (sum of kernel time / first-to-last kernel span of the recorded
    steps -- the rest is the GPU idle, waiting on the CPU), time by category, top kernels."""
    import torch
    kernels = [e for e in prof.events()  # excluding the phase ranges' GPU-side annotations
               if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in PHASES]
    busy = sum(e.time_range.elapsed_us() for e in kernels)
    span = max(e.time_range.end for e in kernels) - min(e.time_range.start for e in kernels)
    by_cat, by_name = defaultdict(float), defaultdict(float)
    for e in kernels:
        by_cat[categorize(e.name)] += e.time_range.elapsed_us()
        by_name[e.name] += e.time_range.elapsed_us()
    top = sorted(by_name.items(), key=lambda kv: -kv[1])
    return {"gpu_busy_pct": round(100 * busy / span, 1), "span_ms": round(span / 1000), "n_kernels": len(kernels),
            "by_category_pct": {c: round(100 * t / busy, 1) for c, t in sorted(by_cat.items(), key=lambda kv: -kv[1])},
            "top10_ms": [(n[:100], round(t / 1000, 1)) for n, t in top[:10]],
            "other_top5_ms": [(n[:100], round(t / 1000, 1)) for n, t in top if categorize(n) == "other"][:5]}


def main():
    args = parse_args()
    from training.qlora_financial.train import load_split
    out = Path(args.out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    ds = load_split(args.data_dir, "train")
    if args.mode == "profile":  # experiment A: v1's exact settings on the real length mix
        mix = ds.select(sorted(random.Random(0).sample(range(len(ds)), args.steps * 16)))
        phases = make_phase_callback(nsys_capture=args.nsys)
        if args.nsys:  # nsys records the timeline; this run only reports tokens/s
            result = run_config(args, mix, batch_size=1, grad_accum=16, callbacks=[phases])
        else:  # profiler callback listed first, so it steps before a new phase range opens
            cb = make_profiler_callback(str(out / "traces" / "profile_v1_trace.json"))
            result = run_config(args, mix, batch_size=1, grad_accum=16, callbacks=[cb, phases])
            result["profile"] = summarize_profile(cb.prof)
            result["phases"] = phase_breakdown(cb.prof)
        (out / ("profile_v1_nsys_run.json" if args.nsys else "profile_v1.json")).write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    else:  # experiments B and C
        rows = []
        lengths = token_lengths(ds, args.model)
        for label, sub in bucket_by_length(ds, lengths, args.steps * 16).items():  # B: batch 1 per length bucket
            rows.append({"experiment": "B", "data": label, **run_config(args, sub, 1, 16)})
            print(rows[-1])
        mix = ds.select(sorted(random.Random(0).sample(range(len(ds)), args.steps * 16)))  # same mix as A
        # C: effective batch stays 16. Random order is the same across batch sizes (same seed), so
        # those final losses should match; grouped changes the order, and batch 1 has no padding to save.
        for bs, sampling in [(1, "random"), (2, "random"), (2, GROUPED), (4, "random"), (4, GROUPED),
                             (8, "random"), (8, GROUPED)]:
            rows.append({"experiment": "C", "data": "mix", **run_config(args, mix, bs, 16 // bs, sampling)})
            print(rows[-1])
        (out / "results.json").write_text(json.dumps(rows, indent=1))
        with (out / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
