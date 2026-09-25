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
    return SFTConfig(
        output_dir=str(Path(args.output_dir) / ("smoke" if args.smoke else "full")),
        model_init_kwargs={"dtype": torch.bfloat16},  # TRL's default is float32
        max_length=args.max_length,  # TRL's default 1024 would cut answers off
        num_train_epochs=1, max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        per_device_eval_batch_size=4,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=50,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps", eval_steps=every,
        save_strategy="steps", save_steps=every, save_total_limit=2,
        report_to="mlflow", run_name="smoke" if args.smoke else "full",
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
    config = sft_config(args)
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=train_ds,
                         eval_dataset=val_ds, quantization_config=quant, peft_config=lora)
    trainer.train()
    adapter_dir = Path(config.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))  # LoRA adapter only, not a merged model
    print(f"Saved adapter to {adapter_dir}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
