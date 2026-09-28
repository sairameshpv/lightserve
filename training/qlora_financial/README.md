# QLoRA fine-tune of Llama-3-8B-Instruct on financial-document QA

This is the project's first *training* round. Every earlier round was
inference-serving: lightserve itself and the vLLM benchmarks. It fine-tunes
`meta-llama/Meta-Llama-3-8B-Instruct` with QLoRA (a 4-bit NF4 frozen base
plus trainable LoRA adapters) on ~34k real, document-grounded financial QA
examples from four academic datasets. It uses one preemptible L40S, is
tracked in MLflow, and touches no lightserve code.

**Result**: on 200 held-out numeric test questions, from documents never
seen in training, accuracy went from **31.5% (original) to 74.5%
(fine-tuned)**. Part of that gap is answer *format*: the original model
writes long explanations, and the scorer reads the last number. Even when
the original model is scored as correct whenever the right number appears
*anywhere* in its answer (47.5%, a generous upper bound), the fine-tuned
model is **at least +27 points** ahead. Validation loss fell at every one
of 9 checks (0.596 → 0.441) and flattened near the end of the single
epoch, with no sign of memorizing. It isn't a solved problem: the
fine-tuned model still makes arithmetic slips and can give a short,
confident, invented answer (see *Reading this*).

## Data

`prepare_dataset.py` turns four sources, all loaded from their *train*
splits only, into one chat format (system / user / assistant):

| Source | Examples | Target format |
|---|---|---|
| `virattt/financial-qa-10K` | 7,000 | full-sentence answer, as-is |
| `FinGPT/fingpt-convfinqa` | 11,104 | bare number (e.g. `206588.0`), kept as-is; rewriting it as a sentence would mean inventing text |
| FinQA (`wandb/finqa-data-processed`) | 6,624 | gold `program` rendered as `Step N: ...` lines, then `Answer: X` |
| `next-tat/TAT-QA` | 13,251 | step chosen by `answer_type`: compute (arithmetic), count (`##` items), compare (`a > b`); annotator notes like "locate and analyze ... in row 4" dropped (answer only) |

That's 37,979 examples, with 41 exact duplicates dropped, leaving **37,938**.

**The split is by source document, not by question.** A random split
leaked test content into training in two ways, both checked on the real data:
ConvFinQA stores one row per conversation turn (11,104 rows over 1,588
documents, up to 24 turns), and each later turn's prompt repeats earlier
turns' *answers*; and ConvFinQA was built from FinQA's own reports. Every
example carries a document group (10-K passage, ConvFinQA document, FinQA
page, TAT-QA table). ConvFinQA documents sharing at least 2 long sentences
with a FinQA page are merged into that page's group, which linked 1,550 of
1,588. Whole groups are then assigned to one split: **34,141 / 1,899 /
1,898** (90 / 5 / 5%), with an assert that no group crosses splits.

**Length**: p50 356 tokens, p99 1,876, max 3,197 (21.05M training tokens).
264 examples exceed 2,048. Truncation cuts the *end*, which holds the
question and the answer, so `max_length` is **3,200** and every example is
kept whole.

## Setup

| | |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct`, 4-bit NF4 + double quantization, bf16 compute |
| LoRA | r=16, alpha=32, dropout 0.05, all 7 projections (q/k/v/o, gate/up/down); adapter 84 MB |
| Training | 1 epoch, batch 1 × grad accumulation 16 = 2,134 steps, lr 2e-4 cosine, 50 warmup steps, `max_length` 3,200 |
| Loss | answer only: conversational prompt-completion format (TRL computes loss on the completion); `assistant_only_loss` would need `{% generation %}` markers in the chat template |
| Eval / checkpoints | full 1,899-example val set and a checkpoint every 250 steps; `--resume` continues from the newest checkpoint |
| Hardware | one **preemptible** L40S (46 GB usable), Nebius; not reclaimed during the run |
| Software | torch 2.11.0+cu128, transformers 5.17, trl 1.14, peft 0.21, bitsandbytes 0.50, mlflow 3.16 (SQLite store: `outputs/mlflow.db`) |

```
python3 -m training.qlora_financial.prepare_dataset   # writes data/{train,val,test}.jsonl
python3 -m training.qlora_financial.train --smoke     # 32 longest examples, 2 steps (~1 min)
python3 -m training.qlora_financial.train             # full run
python3 -m training.qlora_financial.verify_finetune   # original vs fine-tuned on test
```

Run on the node as `sudo HF_HUB_OFFLINE=1 <venv>/bin/python -m ...`,
because the checkpoint cache is under `/root`. Problems actually hit:
- **Default PyTorch didn't see the GPU.** `pip` installed torch 2.14.0+cu130,
  but driver 570 supports up to CUDA 12.8. Fix: `pip install torch==2.11.0
  --index-url https://download.pytorch.org/whl/cu128` (the newest cu128 build).
- **The node hung once**: it showed `RUNNING` but SSH was dead during the
  reinstall. `instance stop` + `start` fixed it, and the previous boot's
  log showed no cause.
- **`train.log` doesn't show losses live.** Python block-buffers stdout
  through `tee`, while tqdm writes to stderr unbuffered. Read live
  metrics from `outputs/mlflow.db` instead.
- `vllm-server` auto-starts on boot and holds ~42 GB, so `sudo docker stop
  vllm-server` first and `docker start` it again before stopping the node.

## Results

**Training**: 2,134 steps in **4h52m** (17,540 s including the 9 eval
passes of ~203 s each), 21.05M tokens, **~1,200 tokens/s** on average
(the smoke run's longest examples reached ~2,100). Average train loss
0.489; peak GPU memory **7.4 GiB** of 46 GB.

| Step | 250 | 500 | 750 | 1,000 | 1,250 | 1,500 | 1,750 | 2,000 | 2,134 |
|---|---|---|---|---|---|---|---|---|---|
| Eval loss | 0.596 | 0.572 | 0.523 | 0.495 | 0.469 | 0.450 | 0.442 | 0.441 | **0.441** |
| Eval token accuracy | 86.1% | 86.7% | 87.5% | 88.2% | 88.8% | 88.9% | 89.1% | 89.2% | **89.2%** |

**Verification** (`verify_finetune.py`): 200 test questions whose correct
answer is a single number, fixed seed. Both models run from the same 4-bit
weights, with the adapter on vs. off. The scoring rule reads the last number
after the last `Answer:` (or in the whole text), within 1% or ×100/÷100
(percent vs. fraction).

| Dataset | n | Original | Fine-tuned |
|---|---|---|---|
| ConvFinQA | 89 | 28.1% | **84.3%** |
| FinQA | 46 | 28.3% | **63.0%** |
| TAT-QA | 58 | 34.5% | **65.5%** |
| financial-qa-10K | 7 | 71.4% | 100.0% |
| **All** | **200** | **31.5%** | **74.5%** |

Outcomes: 94 fine-tuned-only correct, 55 both, **8 original-only**, 43
neither.

**Format check**: the same answers, re-scored as correct if the right
number appears *anywhere* in the answer:

| | Strict (last number) | Lenient (any number) | Median answer length |
|---|---|---|---|
| Original | 31.5% | 47.5% | 308 chars |
| Fine-tuned | 74.5% | 75.5% | 12 chars |

## Reading this

**The strict score overstates the gain; the honest claim is at least
+27 points.** The scoring rule reads the last number, and the original model
writes explanations. For example, on *"what was the net change?"* (correct:
`-5.7`) it answered *"135.2 − 129.5 = −5.7 million. So, the unrecognized
tax benefits decreased by $5.7 million"*. That's right, but it ends on
positive 5.7, so it's scored wrong. The lenient check removes that penalty,
but it is **generous** to the original model: a 308-character answer
contains many numbers (quoted figures, intermediate results), and any match
counts. So 47.5% is an upper bound on the original model's real accuracy,
and the fine-tuned model's strict 74.5% beats it by at least 27 points. About
16 of the 43 strict points are format.

**What the fine-tuning taught**: the short `Step N: ... / Answer: X` format
(the median answer went from 308 to 12 characters), and usually the right
*method*. On the dilutive-shares question it wrote `divide 2.2 and 169.6`,
equivalent to the gold `171.8 / 169.6 − 1`, and got `0.01297` exactly.

**What it didn't fix:**
- *Arithmetic.* It no longer writes intermediate results, so it computes
  "in its head". On a percentage-change question it wrote exactly the
  correct two steps (`subtract 6348 and 6241`, `divide ... by 6241`) and
  still answered 0.01801 instead of 0.01714. The original model, writing
  out `107 / 6241`, got it right. Of the 8 original-only wins, 3 are this
  (right method, result off by more than 1%), 3 are real method errors
  (wrong formula, a missed ×1,000,000 unit step, a wrong value), and **2 are
  scorer artifacts against the fine-tuned model**: it answered `$5,910
  thousand` / `782 thousand` (TAT-QA's scale word, learned from training)
  where the stored answer is `$5,910` / `782`, and the scorer multiplies the
  scale out.
- *Reading the table.* On one question both models divided by the same wrong
  figure (28,383 instead of 18,988). The fine-tuned model's arithmetic was
  right for the wrong input.
- *Confident invention.* One TAT-QA sentence answer was made up ("the sale
  of certain non-core businesses...") and stated as briefly and confidently
  as a correct one. The short format hides the working that exposed such
  errors in the original model's answers.

**Limits of this measurement**: one training run and one seed; 200 scored
questions (only 7 from financial-qa-10K, so that row is noise); numeric
answers only, while sentence answers were only read, not scored; no check of
general ability (catastrophic forgetting); and the scorer's ×100 leniency
can rarely accept a wrong answer. Efficiency is also unoptimized: the GPU
spot check showed **37%** utilization, and peak memory was 7.4 of 46 GB.
Batch size 1 leaves most of the L40S idle on short examples.

## What's next

- **Throughput**: larger per-device batches (grouped by length or packed)
  and 2-GPU data parallel, measured against this run as the 1-GPU,
  batch-1 baseline (4h52m, ~1,200 tokens/s, 37% utilization).
- **Evaluation v2**: serve the adapter through vLLM (`--enable-lora`) and
  compare both models with promptfoo; an MLflow LLM judge for the sentence
  answers; the datasets' official test sets via FinBen, which were never used
  here; a general-ability check such as an MMLU subset, for forgetting.
- **Scorer fixes**: treat a matching number with a scale word
  (`782 thousand` vs `782`) as correct, and add the lenient any-number
  score to `verify_finetune.py` itself (for this README it was computed
  separately, from `verify_results.jsonl`).
- **Arithmetic**: a calculator or tool step, so the model emits the
  formula and the result is computed exactly.
- **Serving on lightserve**: merge the adapter into bf16 weights
  (`merge_and_unload()`), then load it with lightserve's own loader.
  lightserve has no LoRA or 4-bit support today.

## Files

- `prepare_dataset.py`: loads the four sources, formats, deduplicates, links
  documents, splits by group, runs the length audit, writes `data/*.jsonl`.
- `train.py`: QLoRA training (`--smoke`, `--resume`), MLflow tracking,
  saves the adapter and prints peak memory.
- `verify_finetune.py`: original vs fine-tuned on `test.jsonl`, readable
  side-by-side plus number-scored, writes `verify_results.jsonl`.
- `requirements.txt`: dependency floors; on a CUDA 12.8 driver, install
  torch from the cu128 index (see *Setup*).

Not committed (`.gitignore`): `data/` (regenerated identically by
`prepare_dataset.py`, fixed seed) and `outputs/` (the adapter, checkpoints,
`mlflow.db`, `verify_results.jsonl`, logs). To reproduce, run the four
commands in *Setup* in order. View the runs with
`mlflow ui --backend-store-uri sqlite:///training/qlora_financial/outputs/mlflow.db`.
