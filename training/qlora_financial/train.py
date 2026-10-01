"""QLoRA fine-tune of meta-llama/Meta-Llama-3-8B-Instruct on the
financial-QA set written by prepare_dataset.py, tracked in MLflow.

Needs an NVIDIA GPU (4-bit loading via bitsandbytes) -- run on the L40S.
Like prepare_dataset.py, the heavy imports (torch, peft, trl) live inside
the functions that need them, so the data-loading half still runs on a
machine with only `datasets` installed.

max_length is 3200, not TRL's default 1024: the longest real example is
3,197 tokens (prepare_dataset.py's length audit), and truncation cuts the
*end* -- the question and answer -- so every example is kept whole.

Run (always --smoke first: trains on the 32 longest examples, so the
worst-case memory at 3200 is hit in minutes, not hours into the real run):
    python3 -m training.qlora_financial.train --smoke
    python3 -m training.qlora_financial.train
"""

import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="training/qlora_financial/data")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="HF model id, or a local checkpoint folder on the node")
    ap.add_argument("--output-dir", default="training/qlora_financial/outputs")
    ap.add_argument("--max-length", type=int, default=3200)
    ap.add_argument("--smoke", action="store_true",
                    help="32 longest train examples, 2 optimizer steps, separate MLflow experiment")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in the output folder "
                         "(the node is preemptible -- a reclaim loses at most ~250 steps)")
    # Opt-in changes from profiling/report.md's recommendations; defaults = v1 exactly.
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16, help="keep batch-size x grad-accum = 16")
    ap.add_argument("--group-by-length", action="store_true", help="batch similar lengths (less padding)")
    ap.add_argument("--bf16-base", action="store_true", help="plain LoRA: base in bf16, no 4-bit (no dequantize)")
    ap.add_argument("--bf16-adapters", action="store_true",
                    help="cast LoRA adapters to bf16 (PEFT upcasts them to fp32 unless the base is 4-bit)")
    ap.add_argument("--liger", action="store_true", help="Liger fused kernels (pip install liger-kernel)")
    ap.add_argument("--run-name", default="full", help="output subfolder + MLflow run name (v1 = full)")
    return ap.parse_args()


def load_split(data_dir: str, name: str, longest: int = None):
    """One JSONL split as a conversational prompt-completion dataset:
    prompt = [system, user], completion = [assistant]. TRL computes the
    loss on the completion only for this format, so the model is trained
    on answers, not on reproducing the report text. `source`/`group` are
    dropped (bookkeeping, not model input). longest=N keeps only the N
    longest examples (by character count, a cheap proxy for tokens).
    """
    from datasets import load_dataset
    ds = load_dataset("json", data_files=str(Path(data_dir) / f"{name}.jsonl"), split="train")
    ds = ds.map(lambda ex: {"prompt": ex["messages"][:2], "completion": ex["messages"][2:]},
                remove_columns=ds.column_names)
    if longest:
        sizes = [len(p[1]["content"]) for p in ds["prompt"]]
        ds = ds.select(sorted(range(len(ds)), key=lambda i: -sizes[i])[:longest])
    return ds


def qlora_configs():
    """The canonical QLoRA recipe: frozen base weights stored in 4-bit
    NF4 (double-quantized scales save ~0.4 bits/param more), matmuls run
    in bf16; small trainable LoRA adapters on all 7 Llama projection
    layers -- attention (q/k/v/o) and MLP (gate/up/down).
    """
    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    return quant, lora


def sft_config(args):
    """Batch size 1 x 16 accumulation = effective batch 16 (~2,134
    optimizer steps for one epoch); batch size 1 means the 3,200-token
    examples never pad their short neighbours. Smoke mode: 2 steps, eval
    and save after every step, so every code path runs once in minutes.
    """
    import torch
    from trl import SFTConfig
    every = 1 if args.smoke else 250
    # getattr defaults = v1, so callers built before these options (profile_train.py) still work.
    run_name = "smoke" if args.smoke else getattr(args, "run_name", "full")
    grouped = getattr(args, "group_by_length", False)
    return SFTConfig(
        output_dir=str(Path(args.output_dir) / run_name),
        model_init_kwargs={"dtype": torch.bfloat16},  # TRL's default is float32
        max_length=args.max_length,  # TRL's default 1024 would cut answers off
        num_train_epochs=1, max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "grad_accum", 16),
        train_sampling_strategy="group_by_length" if grouped else "random",
        use_liger_kernel=getattr(args, "liger", False),
        per_device_eval_batch_size=4,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=50,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps", eval_steps=every,
        save_strategy="steps", save_steps=every, save_total_limit=2,
        report_to="mlflow", run_name=run_name,
    )


def main():
    args = parse_args()
    import os
    import torch
    from trl import SFTTrainer

    # MLflow's file store (./mlruns) is deprecated since 3.6; SQLite is the
    # current default. One db for smoke + full runs, separate experiments.
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{out / 'mlflow.db'}")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME",
                          "qlora-financial-smoke" if args.smoke else "qlora-financial")

    train_ds = load_split(args.data_dir, "train", longest=32 if args.smoke else None)
    val_ds = load_split(args.data_dir, "val", longest=16 if args.smoke else None)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    quant, lora = qlora_configs()
    if args.bf16_base:
        quant = None  # plain LoRA: base loads in bf16 via sft_config's model_init_kwargs
    config = sft_config(args)
    print(f"base={'bf16 (LoRA)' if args.bf16_base else '4-bit NF4 (QLoRA)'} "
          f"batch={config.per_device_train_batch_size}x{config.gradient_accumulation_steps} "
          f"order={config.train_sampling_strategy} out={config.output_dir} "
          # 4-bit base: TRL itself casts adapters to bf16; bf16 base: PEFT keeps fp32 unless --bf16-adapters
          f"adapters={'bf16' if args.bf16_adapters or not args.bf16_base else 'fp32'} "
          f"liger={'on' if config.use_liger_kernel else 'off'}")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=train_ds,
                         eval_dataset=val_ds, quantization_config=quant, peft_config=lora)
    if args.bf16_adapters:  # TRL's own QLoRA-path cast, before train() builds the optimizer
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    adapter_dir = Path(config.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))  # LoRA adapter only, not a merged model
    print(f"Saved adapter to {adapter_dir}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
