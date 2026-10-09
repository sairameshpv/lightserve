# Continued pretraining: Llama-3-8B-Instruct reads 10-K reports

**What this is.** A first "continued pretraining" run: the chat model we fine-tune elsewhere in this
repo (Llama-3-8B-Instruct) simply **reads** 25 million tokens of SEC 10-K annual-report text, about
1.6% of the library built in `../corpus_prep/`, so it becomes more fluent in financial writing. It
learns through small add-on layers (LoRA), the same kind our fine-tune uses, while the model itself
stays frozen. We then measure, before and after, how well it predicts text it never read, and
whether it is still a good chat model.

**Bottom line.** Reading made 10-K text clearly more predictable for the model (perplexity
−22.7%), but it did **not** make the finance assistant better: our usual fine-tune, redone on top of
the reading model, scored **74.0%** on the same 200 test questions where the original fine-tune (v2,
`../qlora_financial/`) scored **75.0%**, a difference well within run-to-run noise. At this scale
(25M tokens, small add-on layers, one run each), continued pretraining was not worth its cost for
this task. Details in *Did reading help the finance assistant?* below.

## The result

Measured on text the model never trained on: 200 pages (819,200 tokens) from Corpus-Prep's
held-out **check pile**, and the 70-page WikiText-2 test set (ordinary Wikipedia articles).
"Loss" is the model's average surprise at each next token; "perplexity" is e^loss, roughly how
many options it hesitates between. **Lower is better.**

| Text | Before (loss / perplexity) | After (loss / perplexity) | Perplexity change |
|---|---|---|---|
| **10-K reports (check pile)** | 1.5748 / 4.83 | **1.3170 / 3.73** | **−22.7%** |
| Wikipedia (WikiText-2 test) | 2.0473 / 7.75 | 1.9138 / 6.78 | −12.5% |

So on unseen annual-report text the model now hesitates between about 3.7 likely next tokens
instead of 4.8. And it did **not** get worse at ordinary text: it got better too (next section).

**Still a chat model.** Five fixed questions, answered greedily (always the most likely word) before
and after, in `results/before.json` and `results/after.json`: the answers are still correct and
well formed. "What is the capital of Australia?" gets the identical answer (Canberra). The other
four are reworded: the revenue-growth answer still computes 25% (now opening with "A simple one!"),
the haiku is still 5-7-5, and the operating vs. net income and job-interview answers are still
clear and structured (answers stop at 120 tokens, so the long ones end mid-sentence, before and
after alike). Five questions are a sanity check, not a measurement of chat quality.

## What this does and doesn't show

**The surprise: Wikipedia got easier too.** Reading 10-K reports was expected to leave general
text unchanged or slightly worse ("forgetting"). Instead WikiText perplexity fell 12.5%. A likely
explanation, **not tested**: the Instruct model was trained mostly to chat, which leaves it a little
out of practice at predicting plain running text; 25M tokens of plain prose, on any topic, give
some of that practice back. If so, part of the 10-K gain is that same general recovery, not
financial knowledge.

**There is still a 10-K-specific part.** The 10-K loss fell about twice as much as the Wikipedia
loss (0.2578 vs. 0.1335). But "10-K-specific" is broader than "knows more finance": the check pile
comes from the same collection as the reading, so the model also learned that collection's style
(section headings like "Item 7.", the standard legal phrasing, how this dataset lays out text).

**How the two could be separated (not done, and now less important):** a control run, where the
same model reads 25M tokens of general text with identical settings (from a general web collection,
not Wikipedia, so the WikiText check stays fair). The difference on the 10-K check pile would be the
10-K-specific gain (finance knowledge and this collection's style together, which this kind of
measurement can't pull apart). But the test that matters, the fine-tune redone on top, showed no
gain in answers (next section), so this control would only explain a perplexity gain that didn't
reach the task.

## Did reading help the finance assistant?

**How it was tested.** The reading add-on was folded into the model's weights (`merge.py`). A check
confirmed nothing changed: the merged model's 10-K loss was 1.3169 vs. 1.3170 with the add-on
attached, and all 5 chat answers were word-for-word identical (`results/merged_check.json`). Then
v2's exact fine-tuning command ran on the merged model (same data, seed, settings, 2,134 steps), and
`verify_finetune.py` scored it on the same 200 numeric test questions with the same strict scorer.
The only difference from v2 is the starting model.

| | v2 (from the original model) | **New (from the reading model)** |
|---|---|---|
| **Accuracy, all 200** | 75.0% (150) | **74.0% (148)** |
| ConvFinQA (89) | 84.3% | 84.3% |
| FinQA (46) | 67.4% | 63.0% |
| TAT-QA (58) | 63.8% | 63.8% |
| financial-qa-10K (7) | 100% | 100% |
| Final eval loss | 0.425 | 0.4296 |
| Training time (incl. 9 eval passes) | 3 h 00 min | 2 h 59 min |

**Question by question:** 12 were right only in v2, 10 only in the new model, and 115 of the 200
answers were word for word the same. A sign test on the 22 disagreements gives p = 0.83: no
evidence of any difference, the same kind of back-and-forth as v1 vs. v2 (11 vs. 12).

**Reading alone, with no fine-tune.** The same scoring also covers the two starting models: the
read-only model got 35.5% (71) vs. the original Instruct model's 32.0% (64), with 20 questions right
only after reading and 13 only before (p = 0.30): a slight hint at most, not a measured gain. (It
lost 2 of the 7 easy 10-K questions while gaining on the other three datasets.)

**What it likely means** (interpretations, not tested): the fine-tune, 34,141 question-answer
examples, brings both starting points to the same place, so whatever fluency the reading added
doesn't turn into extra right answers; and predicting report text is a different skill from what
these questions demand, the step-by-step arithmetic where the fine-tuned model still slips
(`../qlora_financial/README.md`). **Conclusion:** for this task, at this scale, the reading step
cost about 3.5 hours of paid GPU time (the reading session, 2026-10-07) for no measurable benefit.

## How it was trained

- **Reading material:** 6,103 "pages" of 4,096 tokens (25.0M tokens) cut from random places in
  Corpus-Prep's 16 study files, fixed by a seed so the run can be repeated. A page may run from the
  end of one report into the next; the end-of-document marker sits between them.
- **Learning:** LoRA add-on layers (rank 16, on all 7 projections of each layer, as in our
  fine-tune), kept in 16-bit (bf16) like the model. 382 updates of 16 pages (~65,500 tokens) each,
  learning rate 1e-4 with a 20-update warm-up and a cosine decay.
- **Speed settings, all measured earlier in `../qlora_financial/profiling/report.md`:** bf16
  add-ons, Liger fused kernels, and gradient checkpointing switched off for 8 of the 32 layers.

**Speed and cost.** One L40S, **2 h 35 min** (9,309 s by the trainer's own clock) for 24,997,888
tokens (6,103 pages): **2,685 tokens/s**, steady at about 24.4 s per update from start to finish.
Peak GPU memory **31.7 GiB** of 44.4. That's comparable to our best fine-tuning setting (2,579
tokens/s with FlashAttention; the token mixes differ), with no special attention setup: these pages
have **no padding**, so there is no mask and PyTorch picks its fast flash kernel by itself. A
forward-pass profile of one page confirmed it: `pytorch_flash::flash_fwd_kernel`, 32 times (once
per layer), and no memory-efficient kernels. (`results/summary.json` says 25,034,752 tokens and
2,688 tokens/s: it counted the last update as 16 pages, but it had only 7.)

**Training loss** (average surprise on the pages being read, logged every 5 updates): 1.59 at the
start, then by quarter of the run 1.406 → 1.352 → 1.335 → 1.334, ending at 1.31. Almost all the
drop happens in the first half; the second half barely moves. That hints that more of the same
reading, at these settings, would add little, but it was not tested.

## Lessons and run notes

1. **No padding, no mask, fast attention for free.** The fine-tune needed extra work (padding-free
   batches plus a downloaded FlashAttention kernel) to escape PyTorch's slower attention kernel.
   Fixed-length pages cut from one long token stream have no padding at all, so the fast kernel is
   chosen automatically.
2. **The right "end of document" marker matters.** The pages contain Llama-3's `<|end_of_text|>`
   between reports, not the Instruct tokenizer's default `<|eot_id|>` ("end of a chat turn"); see
   `../corpus_prep/`.
3. **A safeguard, not a tested fix.** `pretrain.py` calls `enable_input_require_grads()`, the
   standard step when gradient checkpointing runs on a frozen model with LoRA. The run was not
   tried without it.
4. **To see which kernel runs, a forward pass is enough.** My first check script ran one page
   forward *and* backward without checkpointing and ran out of memory. Attention picks its kernel in
   the forward pass, so a forward-only profile answered the question with little memory.
5. **Measure with the trainer's own clock, and count what was really read.** The smoke run's
   summary reported 2,056 tokens/s because my timer included 145 s before training started (cause
   unknown; it didn't recur in the full run, where my timer and the trainer's agreed within 6 s).
   The full run's summary also overcounted tokens (the partial last update, see above).
   `pretrain.py` now uses the trainer's own `train_runtime` and counts pages actually read; that
   fix has not run on a GPU yet.
6. **About 28 idle minutes.** Training ended at 00:20 UTC, but my background check only reported it
   at 00:48 (the cause wasn't found), so the node sat idle and billed for ~28 minutes before the
   "after" measurement and shutdown.

## How to run it

On a GPU node with the Corpus-Prep output in `/home/ubuntu/corpus` (otherwise pass `--corpus-dir`),
from the repository root, with the training Python environment and the HF token loaded (the model is
gated). On our Nebius node: `sudo bash -c "set -a; . /root/hf_token.env; set +a; <venv>/bin/python -m ..."`.

```
# 20 updates (~10 min): check memory and speed first
python3 -m training.cpt_financial.pretrain --smoke
# "before" measurement of the original model (~3 min)
python3 -m training.cpt_financial.evaluate --out before.json
# the full run: 25M tokens, ~2 h 35 min on one L40S
python3 -m training.cpt_financial.pretrain
# "after" measurement with the trained add-on
python3 -m training.cpt_financial.evaluate --adapter training/cpt_financial/outputs/run/adapter --out after.json
# did it help the assistant? fold the add-on in, check the merge, then v2's fine-tune + scoring on it
python3 -m training.cpt_financial.merge --adapter training/cpt_financial/outputs/run/adapter --out /home/ubuntu/cpt_merged
python3 -m training.cpt_financial.evaluate --model /home/ubuntu/cpt_merged --out merged_check.json
python3 -m training.qlora_financial.train --model /home/ubuntu/cpt_merged --run-name v2_cpt --bf16-base \
    --bf16-adapters --batch-size 4 --grad-accum 4 --group-by-length --liger
python3 -m training.qlora_financial.verify_finetune --model /home/ubuntu/cpt_merged \
    --adapter training/qlora_financial/outputs/v2_cpt/adapter --bf16-base
# tests (made-up token files, no GPU): pip install pytest first
python3 -m pytest training/cpt_financial/tests
```

## Files

- `pretrain.py`: the pages (`PageDataset`), the model with bf16 LoRA, the training settings.
- `evaluate.py`: check-pile and WikiText loss, and the five fixed chat questions (`--model` for
  a merged model).
- `merge.py`: folds the reading add-on into the model's weights, for the fine-tune on top.
- `tests/test_pretrain.py`: the pages have the right length and content, the same seed gives the
  same pages, pages only come from the files given, and the page count of a partial last update.
- `results/`: `before.json`, `after.json` (losses and the five answers), `summary.json` (the run's
  speed, memory and every logged loss; note its token overcount, explained above),
  `merged_check.json` (the merged model's losses and answers), `finetune_comparison.json` (v2 vs.
  the new fine-tune and the two starting models: accuracy per dataset, sign tests, and right/wrong
  for each of the 200 questions).
- Not in git: the trained add-on (81 MB), the merged model (15 GB, `/home/ubuntu/cpt_merged`), the
  new fine-tune (`training/qlora_financial/outputs/v2_cpt/`, with its full answers) and the run logs
  are on the Nebius node; the logs and both runs' full answers are also in the Mac's git-ignored
  `outputs/results/`.

## Next steps

The question this work set out to answer is answered: at this scale, reading 10-Ks didn't make the
fine-tuned assistant better. Reasonable options from here:

1. **Stop here** (my recommendation): the cost is real and the measured benefit is none.
2. **If revisited,** change what's being tested rather than repeating it: read far more (e.g. 250M+
   tokens, ten times this run), and/or measure on a task closer to reading, such as answering from long
   10-K passages, where fluency in report text should matter more than in short arithmetic questions.
3. **The control run** (general text instead of 10-Ks) only if the perplexity gain itself becomes
   the question; it is not needed to judge the assistant.
