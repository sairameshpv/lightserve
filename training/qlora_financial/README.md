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
confident, invented answer (see *Reading this*). A profiling-driven **v2**
retrain trained **38% faster (3h00m vs. 4h52m) at the same accuracy** (see *v2*).
A later check on all 1,404 numeric test questions found the "within 1%" scorer flattering:
it accepts real arithmetic slips. Under a fairer scorer v2 gets **60.5%**, and a calculator
that redoes v2's own written steps brings it to **73.8%** with no training (see *How good is
v2, really?*). The comparisons above still hold, since every column used the same scorer.
**v3** retrains v2 with ConvFinQA answers written as steps too, so the calculator can redo
them: v3 + calculator reaches **81.2%** (fair), ConvFinQA 69.3% → 86.5% (see *v3*).

## Data

`prepare_dataset.py` turns four sources, all loaded from their *train*
splits only, into one chat format (system / user / assistant):

| Source | Examples | Target format |
|---|---|---|
| `virattt/financial-qa-10K` | 7,000 | full-sentence answer, as-is |
| `FinGPT/fingpt-convfinqa` | 11,104 | bare number (e.g. `206588.0`), kept as-is; rewriting it as a sentence would mean inventing text. Opt-in `--convfinqa-steps`: calculation turns get `Step N:` lines from the original release's programs (see *v3*) |
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

## v2: 38% faster, same accuracy

Profiling (`profiling/report.md`) found four speedups. v2 retrains with all of them, on the
same data, seed and number of steps (2,134), evaluated on the same 200 test questions:
```
train --run-name v2 --bf16-base --bf16-adapters --batch-size 4 --grad-accum 4 --group-by-length --liger
```

| | v1 | v2 |
|---|---|---|
| Training time (incl. 9 eval passes) | 4h52m (17,535.9 s) | **3h00m (10,825.1 s), −38%** |
| Training-only throughput | 1,341 tok/s | **2,258 tok/s (+68%)** |
| Final eval loss / eval token accuracy | 0.441 / 89.2% | 0.425 / 89.6% |
| **Numeric test accuracy, strict / lenient** | 74.5% / 75.5% | **75.0% / 76.0%** |
| ConvFinQA / FinQA / TAT-QA / 10-K | 84.3 / 63.0 / 65.5 / 100% | 84.3 / 67.4 / 63.8 / 100% |
| Peak GPU memory | 7.4 GiB | 21.7 GiB |

**Reading it:** the speedups cost no accuracy, but v2 is **not "better"**. Per question, 11
were right only in v1 and 12 only in v2, which is run-to-run noise on 200 questions. v2's
eval loss started worse (0.842 vs. 0.596 at step 250), because `group_by_length` trains the
longest batches first, then finished lower. Its "original" column uses the bf16 base (32.0%) vs.
v1's 4-bit base (31.5%), nearly identical. Numbers: `profiling/v2_full_run.json`.

## How good is v2, really?

The 74.5% / 75.0% above come from one answer per question, on 200 questions, scored
"within 1%". This section scores v2 on **all 1,404** numeric test questions, with 17
answers each. It finds that the 1% rule flatters v2 (the **fair** score is **60.5%**),
and that a calculator redoing v2's own written steps recovers most of it (**73.8%**),
with no training. The comparisons above (original vs. fine-tuned, v1 vs. v2) still
hold, because every column used the same rule; it's their absolute level that was high.

### Measuring room to improve (`sample_eval.py`)

Before trying reinforcement learning (GRPO, rewarding correct answers), check that there's
something to reinforce: GRPO can only strengthen answers the model *sometimes* gets right.
So each question gets 1 greedy answer (the usual score) plus 16 sampled ones (temperature 0.7):
- **pass@8**: chance at least one of 8 samples is right (unbiased estimate from 16);
- **maj@8**: the most common number among 8 samples is right (averaged over 100 draws);
- **always / sometimes / never**: right in 16, 1–15, or 0 of the 16 samples.

v2 is served by vLLM 0.26.0 with the LoRA adapter on the bf16 base (how v2 was trained),
on one L40S. 1,404 × 17 = 23,868 answers took **165 s**: answers are short (median 14
characters), vLLM batches hundreds at once, and the 16 samples share one read of the prompt
(prefix caching). **Setup check**: on the 200 questions `verify_finetune.py` scored, vLLM
greedy gets 74.5% vs. the earlier 75.0% (2 questions right only before, 1 only now): the
same model, small engine-level differences.

| Dataset (official 1% checker) | n | greedy | sampled | pass@8 | maj@8 | always | sometimes | never |
|---|---|---|---|---|---|---|---|---|
| ConvFinQA | 579 | 84.3% | 82.0% | 88.4% | 84.0% | 67.4% | 22.1% | 10.5% |
| FinQA | 324 | 66.0% | 57.8% | 75.7% | 63.2% | 30.6% | 48.5% | 21.0% |
| TAT-QA | 452 | 61.9% | 58.0% | 77.1% | 60.4% | 34.1% | 47.3% | 18.6% |
| financial-qa-10K | 49 | 98.0% | 93.5% | 100.0% | 99.0% | 59.2% | 40.8% | 0.0% |
| **All** | **1,404** | **73.4%** | 69.1% | **82.2%** | 72.1% | 47.9% | 37.0% | 15.2% |

73.4% on all 1,404 is the better estimate of v2's official score than 75.0% on 200. Room
(pass@8 − greedy) is **+8.9 points**, mostly TAT-QA (+15.2) and FinQA (+9.7). **Majority
voting doesn't help**: maj@8 is below greedy. v2's single most likely answer is already its best.

### The 1% checker was too lenient (`--checkers`)

Reading the trial answers turned up checker mistakes in both directions. "Within 1%" accepts
**2019 for 2018** (0.05% apart) and **0.7168 for 151/211 = 0.71564** (a real slip, 0.2%
off). It also rejects `782 thousand` for a stored `782`, because it multiplies the scale out.
So `sample_eval.py --checkers` re-scores the saved answers under checker variants, each adding
one change to the one before, and prints 15 random answers each step flips:

| Checker (all 1,404) | greedy | sampled | pass@8 | room | maj@8 | always | sometimes | never | answers flipped |
|---|---|---|---|---|---|---|---|---|---|
| official: within 1% | 73.4% | 69.1% | 82.2% | +8.9 | 72.1% | 47.9% | 37.0% | 15.2% | |
| + **years**: a year must match exactly | 73.1% | 68.9% | 82.2% | +9.1 | 71.9% | 47.3% | 37.5% | 15.2% | 47 |
| + **scale**: forgive an added/dropped scale word | 75.4% | 71.3% | 83.4% | +8.0 | 74.2% | 50.5% | 35.1% | 14.4% | 565 |
| + **fair**: within 0.1%, or honest rounding | **60.5%** | 57.0% | **75.9%** | **+15.3** | 59.2% | 36.0% | 43.7% | 20.4% | 3,494 |

What reading the flipped examples showed:
- **years**: all 15 were real mistakes, e.g. 2018 for 2017, sometimes with an invented
  justification (`compare 7.1>5.1`) for the wrong year.
- **scale**: adding or dropping a word is fair (`782 thousand` for `782`), but *swapping* one
  is 1,000× off (`-40 million` for `-40 thousand`): 171 such answers, so only one-sided
  differences are forgiven.
- **fair**: in the 0.1–1% band, **14 of 15 were real arithmetic slips**, and 1 was honest
  rounding (`39.34 percent` for `39.3 percent`). v2 picks the right operation and numbers,
  then slips at the 3rd–4th digit doing division in its head (`divide 662.3 and 7.3`: 90.516
  vs. 90.726). "Honest rounding" means the *stored* answer was rounded and v2 gave at least as
  many digits (`0.79 percent` for `0.8 percent`); a *coarser* answer must still be within
  0.1%. Two bugs in the first version were caught by reading what it accepted: `-1.0` for
  `0.0` (−1.0 ÷ 100 rounded to 0.0), and `$0` for `$2.0 million` (rounding the precise stored
  answer to the model's coarse one). Both have tests.

Under **fair**, v2 scores **60.5%** and the room is **+15.3** points (43.7% of questions
"sometimes" right). The official checker stays as it is in `verify_finetune.py`; fair is a
judgment call, shown alongside it rather than replacing it.

### A calculator does the arithmetic (`calculator.py`, `--checkers --calculator`)

v2 writes its method in the training format, so a program can redo the maths exactly and
replace the final number, keeping v2's unit word:
- **FinQA** steps (all 324 answers have them): `add / subtract / multiply / divide / raise to
  the power of X and Y`, chained through `the result of step N`; `6.625%` means 0.06625, as in
  FinQA's programs. Other steps (`take the max of ...`) are left alone.
- **TAT-QA** `compute` formulas (281 of 452 greedy answers; 29 more are count / compare
  steps, left alone): evaluated by walking the parsed syntax tree, allowing only numbers,
  `+ - * /` and brackets, so text is never run as code. `(14,299)` is an accounting
  negative. A ratio reported as `39.3 percent` is ×100.
- **ConvFinQA and 10-K answers have no steps** (trained as bare numbers or sentences), so they
  are unchanged.

**Safety check first**: run on the *correct* reasoning of every test question, it reproduces
the stored answer for **310 of 310** calculable FinQA and **272 of 275** TAT-QA questions. The
3 misses are stored answers with float noise from rounding: `0.30000000000000004` for
68,903 / 226,067 = 0.30479 (and 0.69, 0.83 likewise). The calculator is right, and those 3
questions can't be scored under fair.

| Fair checker | n | greedy as written | greedy with calculator | pass@8 with calc. | room left |
|---|---|---|---|---|---|
| ConvFinQA | 579 | 69.3% | 69.3% (no steps) | 82.8% | **+13.5** |
| FinQA | 324 | 42.0% | **74.7% (+32.7)** | 80.4% | +5.7 |
| TAT-QA | 452 | 58.6% | **76.3% (+17.7)** | 85.5% | +9.2 |
| financial-qa-10K | 49 | 98.0% | 98.0% (no steps) | 100.0% | +2.0 |
| **All** | **1,404** | **60.5%** | **73.8% (+13.2)** | 83.7% | +9.9 |

With the calculator, maj@8 (72.8%) is still below greedy. Why v2's wrong greedy answers were wrong:

| | FinQA | TAT-QA |
|---|---|---|
| Arithmetic slip: the calculator fixes it | **107** | **80** |
| Wrong method: calculable, still wrong | 80 | 50 |
| No calculable steps (count / compare / no formula) | 1 | 57 |

Reading 15 random wrong-method answers: 6 used **the wrong numbers from the table** (another
row or year), 3 **subtracted in the wrong order** (`108-437` for `437-108`), 3 used **the wrong
formula** (% change for a percentage-point difference), 1 added an extra item, 1 looped
(`add ... and 4` ten times), and 1 was one of the 3 float-noise questions above, so it was really
right. What's left is mostly *reading the table*, not arithmetic.

**Limits of this measurement**: one sampling run (temperature 0.7, no fixed seed, because
whether vLLM gives `n=16` distinct samples under one seed wasn't verified), so sampled figures
would shift slightly on a rerun; greedy ones should barely move (vLLM batching can still flip
a borderline token). The fair checker is a judgment call, backed by reading 15 examples per
step, which is a small sample. The calculator was applied
offline to saved answers, not yet in serving, and only to the final number: it can't fix a
wrong method, and v2 might write different steps if it knew a calculator would run them. The
3 float-noise questions count as wrong under fair whatever the model says. Numeric questions
only, as before.

## v3: ConvFinQA with steps

After the calculator, ConvFinQA had the most room left (+13.5), because its answers were
trained as bare numbers, so v2 did that arithmetic in its head and the calculator had nothing
to redo. v3 retrains v2 with one change: ConvFinQA *calculation* turns now get `Step N:` lines
like FinQA's.

**Getting the steps** (`prepare_dataset.py --convfinqa-steps`). The FinGPT copy has no
calculation programs. The original ConvFinQA release (github.com/czyssrs/ConvFinQA, MIT) has
one per turn, and its `train_turn.json` has the same 11,104 turns. Rather than switch sources,
which would reshuffle the split and make every earlier number incomparable, each existing row
is matched to its original turn by the conversation's questions (document text breaks ties).
That matches 11,080 of 11,082 rows, and every matched answer agrees. Calculation turns (`cur_type
== program_turn`, 65%) are rendered with FinQA's own `_program_to_steps`; look-up turns
(`181001`) stay bare numbers. Checked before training:
- without the flag, `train/val/test.jsonl` are byte-for-byte unchanged;
- with it, the same rows land in the same splits with the same prompts and groups, and only
  7,186 ConvFinQA answers change, all keeping their final answer;
- the longest example is 3,199 tokens (limit 3,200);
- the calculator reproduces 378 of 378 calculable ConvFinQA test answers from their steps.

**Training**: v2's exact command plus `--data-dir`, on the same L40S: ≈10,770 s (2h59m) vs.
v2's 10,825 s, peak 21.7 GiB. Final eval loss 0.327, token accuracy 91.7%. These aren't
comparable to v2's 0.425 / 89.6%: the step wording ("Step 1: subtract … and …") is easy to
predict and pulls the average down.

**Results** on the same 1,404 questions, scored as in *How good is v2, really?* (vLLM, greedy +
16 samples, 220 s). Fair checker, greedy (pass@8):

| Dataset | v2 | v3 | v2 + calculator | **v3 + calculator** |
|---|---|---|---|---|
| ConvFinQA | 69.3% (82.8) | 73.7% (87.8) | 69.3% (82.8) | **86.5% (91.5)** |
| FinQA | 42.0% (62.6) | 42.0% (65.2) | 74.7% (80.4) | 73.8% (82.2) |
| TAT-QA | 58.6% (73.8) | 60.2% (73.7) | 76.3% (85.5) | 77.9% (85.1) |
| financial-qa-10K | 98.0% (100.0) | 98.0% (99.8) | 98.0% (100.0) | 98.0% (99.8) |
| **All** | **60.5%** (75.9) | **62.9%** (78.5) | **73.8%** (83.7) | **81.2%** (87.6) |

Under the official 1% checker, v3 greedy is 74.1% vs. v2's 73.4% (ConvFinQA 87.4% vs. 84.3%):
that checker already forgave most of the slips the calculator fixes, so it hides most of the gain.

**Same questions, both models** (greedy, with the calculator):

| Dataset | right only with v2 | right only with v3 | sign test p |
|---|---|---|---|
| ConvFinQA | 22 | **122** | 5e-18 |
| FinQA | 15 | 12 | 0.70 |
| TAT-QA | 10 | 17 | 0.25 |
| **All** | 47 | **151** | 6e-14 |

So v3 + calculator is **+7.4 points** overall, all of it from ConvFinQA (+17.3); FinQA and TAT-QA,
whose data didn't change, didn't change. v3 writes steps on 376 of 579 ConvFinQA answers (378
are calculation turns), and writing steps alone already helps a little: ConvFinQA 69.3% → 73.7%
without the calculator. With v3 + calculator, room (pass@8 − greedy) is down to **+6.4**
(+15.3 for v2 without it), "never right" to 11.0%, and maj@8 (81.1%) still doesn't beat greedy.

**The 22 ConvFinQA questions v3 lost** (right with v2 + calculator, wrong with v3 + calculator), read one by one:

| What went wrong | n | Example |
|---|---|---|
| **Right answer, wrong steps: the calculator overwrote it** | 8 | `divide 243.0 and 247.1` (should be 235.5) but `Answer: -0.04694`, which is correct; recomputed from the steps: −0.01659 |
| Wrong number from the table | 8 | `add 81 and 64` for capex 2015 + 2016 (should be 90); 95.7 (the January 1, 2007 balance) for the year-end 2007 balance of 135.2 |
| Subtracted in the wrong order | 2 | `subtract 2.84 and 4.40` → −1.56 for 1.56 |
| Look-up turned into a calculation | 2 | "last year with S&P data?" (2013) → `subtract 2013 and 2008` = 5 |
| Missed a unit step | 1 | `15.5` for 15.5 billion |
| Arguably right | 1 | computed 34.4%; the stored answer is 34.0, apparently rounded in the source |

The first row is new: v3's final number isn't always derived from its own steps. In a
conversation, earlier turns' answers sit in the prompt, so it can sometimes "know" the number
while writing sloppy steps, and then the calculator trusts the steps. It's still a large net
win: on ConvFinQA the calculator **fixed 84 greedy answers and broke 10** (on v2: 187 fixed, 1
broken, all FinQA/TAT-QA). 3 of the 22 also show a notation mix-up: `subtract 100 and 5%` where
the data means 5, which the calculator reads as 0.05. That's the training data's own convention
(`divide 8% and 100` → 0.0008), so it's v3's mistake, not the calculator's.

**Limits**: one training run and one seed; one sampling run (temperature 0.7, no seed); the 22
were read once by one reader. v3 was served under the label `v2` because `sample_eval.py`
requests that model name; the adapter files were v3's. The calculator is still applied offline,
to saved answers.

## What's next

- **Throughput**: *done for 1 GPU*. Batching + length grouping, bf16 base and
  adapters, and Liger brought training to 3h00m (−38%, see *v2*). Still open:
  turning off gradient checkpointing (~22 of 46 GB used) and 2-GPU data
  parallel, measured against v2 as the new 1-GPU baseline.
- **Evaluation v2**: serving the adapter through vLLM (`--enable-lora`) is
  *done* for sampling (see *How good is v2, really?*). Still open: compare
  both models with promptfoo; an MLflow LLM judge for the sentence
  answers; the datasets' official test sets via FinBen, which were never used
  here; a general-ability check such as an MMLU subset, for forgetting.
- **Scorer fixes**: *measured* with the checker variants in `sample_eval.py`
  (scale words, exact years, 0.1% or honest rounding). Still open: making
  the fair checker part of `verify_finetune.py` itself, and the lenient
  any-number score (for this README computed separately, from
  `verify_results.jsonl`).
- **Arithmetic**: *measured*: the calculator lifts v2 from 60.5% to 73.8%
  (fair checker) on saved answers. Still open: running it at serving time.
- **ConvFinQA steps**: *done* (see *v3*): matched to the original release's
  programs without moving the split; v3 + calculator 81.2% (fair).
- **Answer vs. calculator disagreement** (unmeasured idea): v3 sometimes
  writes a right answer with wrong steps, and the calculator then breaks it
  (10 of 94 ConvFinQA changes). When the model's own number and the
  recomputed one disagree, that could flag low confidence or trigger a
  second attempt instead of always trusting the steps.
- **GRPO** (reward = the fair checker, with the calculator applied): aimed
  at the method mistakes that remain (wrong table cell, reversed
  subtraction, look-up vs. calculation, steps that don't match the
  answer). Room with v3 + calculator is only **+6.4** overall, so it's a
  weaker case than ever; v3 would be the starting model.
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
- `sample_eval.py`: greedy + 16 sampled answers per numeric test question
  from a vLLM server (resumes after a crash, `--limit` for a trial run),
  pass@8 / maj@8 / buckets, setup check vs. `verify_results.jsonl`, writes
  `outputs/v2/samples.jsonl` + `sample_eval_summary.json`. `--checkers`
  re-scores the saved answers under the checker variants; with
  `--calculator`, the calculator safety check and v2 with vs. without it.
  No server needed for either.
- `calculator.py`: exact arithmetic for v2's written FinQA steps and
  TAT-QA formulas (`apply_calculator`).
- `tests/`: the scoring and calculator logic, Mac-only
  (`python -m pytest training/qlora_financial/tests/` from the repo root).
- `requirements.txt`: dependency floors; on a CUDA 12.8 driver, install
  torch from the cu128 index (see *Setup*).

Sampling run (on the node, default `vllm-server` stopped first):
```
docker run -d --name vllm-lora --gpus all --shm-size 32g --ipc=host \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v <repo>/training/qlora_financial/outputs/v2/adapter:/adapters/v2:ro \
  -p 8000:8000 vllm/vllm-openai:latest --model meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-lora --lora-modules v2=/adapters/v2 --max-lora-rank 16 --enable-prefix-caching
python3 -m training.qlora_financial.sample_eval --limit 20   # trial
python3 -m training.qlora_financial.sample_eval              # all 1,404 (~3 min)
python3 -m training.qlora_financial.sample_eval --checkers [--calculator]   # anywhere, no GPU
```

v3 (original ConvFinQA `data.zip` from github.com/czyssrs/ConvFinQA, unzipped under
`data/convfinqa_original/`, gitignored):
```
python3 -m training.qlora_financial.prepare_dataset --out-dir <data_v3> \
  --convfinqa-steps training/qlora_financial/data/convfinqa_original/data/train_turn.json
train --run-name v3 --data-dir <data_v3> --bf16-base --bf16-adapters --batch-size 4 --grad-accum 4 --group-by-length --liger
python3 -m training.qlora_financial.sample_eval --data-dir <data_v3> --out training/qlora_financial/outputs/v3/samples.jsonl
python3 -m training.qlora_financial.sample_eval --data-dir <data_v3> --out training/qlora_financial/outputs/v3/samples.jsonl --checkers --calculator
```

Not committed (`.gitignore`): `data/` (regenerated identically by
`prepare_dataset.py`, fixed seed) and `outputs/` (the adapter, checkpoints,
`mlflow.db`, `verify_results.jsonl`, logs). To reproduce, run the four
commands in *Setup* in order. View the runs with
`mlflow ui --backend-store-uri sqlite:///training/qlora_financial/outputs/mlflow.db`.
