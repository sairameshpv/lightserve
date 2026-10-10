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


def pages_read(updates: int, pages_per_update: int, total_pages: int) -> int:
    """Pages a run actually read: full updates, except the last may be partial (6,103 = 381 x 16 + 7)."""
    return min(updates * pages_per_update, total_pages)


MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def lora_config(rank: int = 16):
    """The fine-tune's LoRA settings (all 7 projections, from qlora_financial/train.py) at `rank`,
    with alpha = 2 x rank as there (r=16, alpha 32), so the add-on's scale stays the same."""
    import dataclasses
    from training.qlora_financial.train import qlora_configs
    return dataclasses.replace(qlora_configs()[1], r=rank, lora_alpha=2 * rank)


def build_model(model_name: str = MODEL, rank: int = 16):
    """The bf16 model with LoRA layers from `lora_config(rank)`, kept in bf16: PEFT would otherwise
    upcast them to fp32, and keeping them bf16 measured +17% tokens/s in the profiling report (the
    --bf16-adapters finding)."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model = get_peft_model(model, lora_config(rank), autocast_adapter_dtype=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.bfloat16 for p in trainable), "adapters should stay bf16"
    model.enable_input_require_grads()  # gradient checkpointing with a frozen base needs this
    return model


def make_stop_on_rise_callback(rises: int = 3, sets=("domain", "general")):
    """Stop training once either check set's loss has gone up `rises` checks in a row (with checks
    every 250 updates, 3 = 750 updates of getting worse). Read from the trainer's own log, which a
    --resume restores, so a restart doesn't forget earlier checks. All GPUs see the same scores,
    so they stop together."""
    from transformers import TrainerCallback

    class StopOnRise(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            for name in sets:
                key = f"eval_{name}_loss"
                last = [h[key] for h in state.log_history if key in h][-(rises + 1):]
                if len(last) == rises + 1 and all(b > a for a, b in zip(last, last[1:])):
                    print(f"STOP: {key} rose {rises} checks in a row: {last}", flush=True)
                    control.should_training_stop = True
    return StopOnRise()


def train(model, pages, out_dir: str, max_steps: int = -1, ckpt_skip: int = 8,
          grad_accum: int = 8, warmup: int = 20, eval_sets=None, eval_every: int = 250,
          resume: bool = False):
    """2 pages at a time x `grad_accum` steps x GPUs per update: 16 pages (~65k tokens) with 8 on
    1 GPU or 4 on 2. The fine-tune's measured speed settings: Liger kernels, and checkpointing off
    on the first `ckpt_skip` layers. `eval_sets` ({"domain": pages, "general": pages}) are scored
    every `eval_every` updates, logged as eval_domain_loss / eval_general_loss."""
    from transformers import Trainer, TrainingArguments, default_data_collator
    from training.qlora_financial.train import make_ckpt_skip_callback
    config = TrainingArguments(
        output_dir=out_dir, num_train_epochs=1, max_steps=max_steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=warmup, bf16=True,
        gradient_checkpointing=True, use_liger_kernel=True,
        # 2 GPUs only (ignored on 1): the default True clashes with gradient checkpointing on a
        # PeftModel, as found in qlora_financial/train.py.
        ddp_find_unused_parameters=False,
        # one page at a time: equal shares per GPU with no page counted twice, and the mean of
        # page losses, the same number evaluate.py reports
        eval_strategy="steps" if eval_sets else "no", eval_steps=eval_every, per_device_eval_batch_size=1,
        # a restart point at every check; on 2 machines each keeps its own copy, since --resume
        # looks on its own disk (only the first machine would write one otherwise)
        logging_steps=5, save_steps=eval_every, save_total_limit=2, save_on_each_node=True,
        # the restart point with the best 10-K score is kept too, even if later ones got worse
        metric_for_best_model="eval_domain_loss" if eval_sets else None, greater_is_better=False,
        report_to="mlflow")
    callbacks = [make_ckpt_skip_callback(ckpt_skip)] if ckpt_skip else []
    if eval_sets:
        callbacks.append(make_stop_on_rise_callback())
    trainer = Trainer(model=model, args=config, train_dataset=pages, eval_dataset=eval_sets,
                      data_collator=default_data_collator, callbacks=callbacks)
    trainer.train(resume_from_checkpoint=resume or None)  # True: the latest in out_dir
    return trainer


def main():
    import argparse, json, os
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", default="/home/ubuntu/corpus", help="Corpus-Prep output (train_*.bin)")
    ap.add_argument("--out-dir", default="training/cpt_financial/outputs")
    ap.add_argument("--tokens", type=int, default=25_000_000, help="how much to read")
    ap.add_argument("--page-len", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true", help="20 updates only: check memory and speed")
    ap.add_argument("--lora-rank", type=int, default=16, help="add-on size (alpha = 2 x rank)")
    ap.add_argument("--warmup", type=int, default=20, help="updates to ramp the learning rate up")
    ap.add_argument("--run-name", help="output folder name (default: smoke or run)")
    ap.add_argument("--resume", action="store_true", help="continue from the latest restart point")
    args = ap.parse_args()
    import torch
    out = Path(args.out_dir).resolve() / (args.run_name or ("smoke" if args.smoke else "run"))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{out.parent / 'mlflow.db'}")  # as in train.py
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "cpt-financial-smoke" if args.smoke else "cpt-financial")
    study = sorted(Path(args.corpus_dir).glob("train_*.bin"))
    pages = PageDataset(study, args.page_len, args.tokens // args.page_len, seed=0)
    print(f"{len(study)} study files, {len(pages)} pages of {args.page_len} tokens")
    # 16 pages per update on any number of GPUs: 2 at a time x grad_accum rounds x GPUs (torchrun
    # sets WORLD_SIZE). Worked out here so a forgotten flag can't silently double the update size.
    gpus = int(os.environ.get("WORLD_SIZE", 1))
    assert 16 % (2 * gpus) == 0, f"16 pages per update doesn't split over {gpus} GPUs"
    grad_accum = 16 // (2 * gpus)
    # The same pages evaluate.py scores before/after, so the curve lines up with those numbers.
    # Scoring doesn't change the weights, only adds time (~1-2 min per check).
    from transformers import AutoTokenizer
    from training.cpt_financial.evaluate import check_pages, wikitext_pages
    eval_sets = {"domain": check_pages(args.corpus_dir), "general": wikitext_pages(AutoTokenizer.from_pretrained(MODEL))}
    trainer = train(build_model(rank=args.lora_rank), pages, str(out), max_steps=20 if args.smoke else -1,
                    grad_accum=grad_accum, warmup=args.warmup,
                    eval_sets=eval_sets, eval_every=10 if args.smoke else 250, resume=args.resume)
    if not trainer.is_world_process_zero():
        return  # on 2 GPUs, only the first writes the summary and saves the add-on
    # The trainer's own clock (training only): an outside timer once included 145 s of unexplained
    # set-up. Tokens = pages actually read by all GPUs: the last update can hold fewer pages.
    secs = next(h["train_runtime"] for h in reversed(trainer.state.log_history) if "train_runtime" in h)
    tokens = pages_read(trainer.state.global_step, 16, len(pages)) * args.page_len
    summary = {"updates": trainer.state.global_step, "stopped_early": 0 < trainer.state.global_step < trainer.state.max_steps,
               "tokens": tokens, "seconds": round(secs, 1),
               "tokens_per_s": round(tokens / secs), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "losses": [h["loss"] for h in trainer.state.log_history if "loss" in h]}
    print(json.dumps({k: v for k, v in summary.items() if k != "losses"}), "last losses:", summary["losses"][-3:])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    trainer.save_model(str(out / "adapter"))  # the LoRA add-on only, not a merged model


if __name__ == "__main__":
    main()
