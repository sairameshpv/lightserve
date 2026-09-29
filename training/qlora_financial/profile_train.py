"""Profile QLoRA training: why the v1 run used ~37% of the L40S, and what fixes it.

v1 (train.py, batch 1 x 16 accumulation) ran at ~1,200 tokens/s overall but
~2,100 on the smoke run's longest examples, with peak memory 7.4 of 46 GB --
pointing at short sequences at batch 1 leaving the GPU idle. Three experiments:
  A (--mode profile): torch.profiler over a few v1-setting steps -> GPU time
    by kernel category, GPU busy % (kernel time / wall time), top kernels.
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
    return ap.parse_args()


# Token-length buckets for experiment B (upper bound 3200 = train.py's max_length).
BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 3200)]
# train_sampling_strategy value for length-grouped batches -- NOT yet confirmed
# against the installed transformers; check its accepted values on the node first.
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
    ("elementwise/norm", ["elementwise", "vectorized", "reduce", "norm", "silu", "softmax", "index", "cat"]),
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
                wait=wait, warmup=warmup, active=active))
            self.prof.__enter__()

        def on_step_end(self, args, state, control, **kw):
            self.prof.step()

        def on_train_end(self, args, state, control, **kw):
            self.prof.__exit__(None, None, None)
            self.prof.export_chrome_trace(trace_path)
    return ProfilerCallback()


def summarize_profile(prof) -> dict:
    """GPU busy % (sum of kernel time / first-to-last kernel span of the recorded
    steps -- the rest is the GPU idle, waiting on the CPU), time by category, top kernels."""
    import torch
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
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
        cb = make_profiler_callback(str(out / "traces" / "profile_v1_trace.json"))
        result = run_config(args, mix, batch_size=1, grad_accum=16, callbacks=[cb])
        result["profile"] = summarize_profile(cb.prof)
        (out / "profile_v1.json").write_text(json.dumps(result, indent=1))
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
