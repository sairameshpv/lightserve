"""Continued pretraining: Llama-3-8B-Instruct reads part of the Corpus-Prep library (SEC 10-K
text, training/corpus_prep/) with LoRA, so it gets more fluent in financial writing.

The library is flat uint32 token files with <|end_of_text|> between reports. Training reads
fixed-length "pages" (windows) from random places in them.
"""

import numpy as np


class PageDataset:
    """`num_pages` windows of `page_len` tokens, from random places across `paths` (a file's chance
    is proportional to its size), fixed by `seed` so a run can be repeated. Files are memory-mapped:
    nothing is read until a page is asked for. A page may run across the end of one report into the
    next; the end-of-document token between them marks the boundary (the standard approach)."""

    def __init__(self, paths, page_len: int, num_pages: int, seed: int):
        self.files = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
        self.page_len = page_len
        room = np.array([len(f) - page_len + 1 for f in self.files])  # possible start positions
        rng = np.random.default_rng(seed)
        self.which = rng.choice(len(self.files), size=num_pages, p=room / room.sum())
        self.start = rng.integers(0, room[self.which])

    def __len__(self):
        return len(self.which)

    def __getitem__(self, i):
        s = self.start[i]
        ids = np.asarray(self.files[self.which[i]][s:s + self.page_len], dtype=np.int64)
        return {"input_ids": ids, "labels": ids.copy()}  # the model shifts labels by one itself


MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def build_model(model_name: str = MODEL):
    """The bf16 model with the fine-tune's LoRA layers (r=16, alpha 32, all 7 projections, from
    qlora_financial/train.py), kept in bf16: PEFT would otherwise upcast them to fp32, and keeping
    them bf16 measured +17% tokens/s in the profiling report (the --bf16-adapters finding)."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from training.qlora_financial.train import qlora_configs
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model = get_peft_model(model, qlora_configs()[1], autocast_adapter_dtype=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.bfloat16 for p in trainable), "adapters should stay bf16"
    model.enable_input_require_grads()  # gradient checkpointing with a frozen base needs this
    return model


def train(model, pages, out_dir: str, max_steps: int = -1, ckpt_skip: int = 8):
    """16 pages (~65k tokens) per update: 2 at a time x 8 accumulation steps. The fine-tune's
    measured speed settings: Liger kernels, and checkpointing off on the first `ckpt_skip` layers."""
    from transformers import Trainer, TrainingArguments, default_data_collator
    from training.qlora_financial.train import make_ckpt_skip_callback
    config = TrainingArguments(
        output_dir=out_dir, num_train_epochs=1, max_steps=max_steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=20, bf16=True,
        gradient_checkpointing=True, use_liger_kernel=True,
        logging_steps=5, save_steps=100, save_total_limit=2, report_to="mlflow")
    callbacks = [make_ckpt_skip_callback(ckpt_skip)] if ckpt_skip else []
    trainer = Trainer(model=model, args=config, train_dataset=pages,
                      data_collator=default_data_collator, callbacks=callbacks)
    trainer.train()
    return trainer


def main():
    import argparse, json, os, time
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", default="/home/ubuntu/corpus", help="Corpus-Prep output (train_*.bin)")
    ap.add_argument("--out-dir", default="training/cpt_financial/outputs")
    ap.add_argument("--tokens", type=int, default=25_000_000, help="how much to read")
    ap.add_argument("--page-len", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true", help="20 updates only: check memory and speed")
    args = ap.parse_args()
    import torch
    out = Path(args.out_dir).resolve() / ("smoke" if args.smoke else "run")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{out.parent / 'mlflow.db'}")  # as in train.py
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "cpt-financial-smoke" if args.smoke else "cpt-financial")
    study = sorted(Path(args.corpus_dir).glob("train_*.bin"))
    pages = PageDataset(study, args.page_len, args.tokens // args.page_len, seed=0)
    print(f"{len(study)} study files, {len(pages)} pages of {args.page_len} tokens")
    model = build_model()
    t0 = time.perf_counter()  # timed: training only, not model loading
    trainer = train(model, pages, str(out), max_steps=20 if args.smoke else -1)
    secs = time.perf_counter() - t0
    tokens = trainer.state.global_step * 16 * args.page_len
    summary = {"updates": trainer.state.global_step, "tokens": tokens, "seconds": round(secs, 1),
               "tokens_per_s": round(tokens / secs), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "losses": [h["loss"] for h in trainer.state.log_history if "loss" in h]}
    print(json.dumps({k: v for k, v in summary.items() if k != "losses"}), "last losses:", summary["losses"][-3:])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    trainer.save_model(str(out / "adapter"))  # the LoRA add-on only, not a merged model


if __name__ == "__main__":
    main()
