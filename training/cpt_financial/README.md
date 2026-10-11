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

**Follow-up (2026-10-11).** A second run read 8× more (200M tokens) with a 4× bigger add-on, on two
GPUs: 10-K perplexity fell further, 3.73 → **3.40** (−29.6% vs. the original model), while Wikipedia
barely moved (6.78 → 6.70), so the extra gain is almost all 10-K-specific. Each extra stretch of
reading helped less than the one before. Whether this helps the finance assistant is **not tested
yet** (it needs the fine-tune on top). Details in *Follow-up: 8× more reading* below.

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

## Follow-up: 8× more reading

**Why.** The first run was small: 25M tokens through a rank-16 add-on. Maybe the model just hadn't
read enough, or the add-on was too small to hold what it read. So a second run (2026-10-10/11)
read **200M tokens** (8×) through a **rank-64** add-on (4×, alpha 128, same 2:1 ratio as before),
on **two GPUs**. Everything else stayed the same: 16 pages per update (3,052 updates), learning rate
1e-4 with a cosine decay (warm-up 100 updates instead of 20, to suit the longer run). The model was
scored on both check sets **every 250 updates while training**, to catch forgetting as it happened.

| Text (perplexity, lower is better) | Original | 25M run (rank 16) | **200M run (rank 64)** |
|---|---|---|---|
| **10-K reports (check pile)** | 4.83 | 3.73 (−22.7%) | **3.40 (−29.6%)** |
| Wikipedia (WikiText-2 test) | 7.75 | 6.78 (−12.5%) | 6.70 (−13.5%) |

Same pages as before, so the numbers compare directly (losses: 10-K 1.2245, Wikipedia 1.9020;
`results/run_200m_r64/after.json`). Against the 25M run, 10-K perplexity fell another 8.8%, while
Wikipedia fell only 1.2%.

**Check by check** (loss, lower is better; each 250 updates is ~16.4M tokens;
`results/run_200m_r64/eval_history.json`). "Learning rate" is the step size as a share of its peak:

| Update | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 | 2500 | 2750 | 3000 | 3052 (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **10-K** | 1.3066 | 1.2861 | 1.2727 | 1.2595 | 1.2487 | 1.2406 | 1.2340 | 1.2286 | 1.2260 | 1.2248 | 1.2245 | 1.2245 | 1.2245 |
| Wikipedia | 1.9082 | 1.9044 | 1.9062 | 1.9052 | 1.9029 | 1.9028 | 1.9035 | 1.9030 | 1.9020 | 1.9018 | 1.9018 | 1.9020 | 1.9018 |
| Learning rate | 99% | 96% | 89% | 79% | 67% | 54% | 41% | 28% | 17% | 8% | 3% | 0% | 0% |

- **The 10-K gain slowed down:** −0.047 from update 250 to 1000, −0.031 from 1000 to 2000, −0.004
  after that. The flat end (1.2245 three times) is mostly the learning rate reaching zero, as the
  schedule intends: the model barely changes then, whatever it reads. So this run can't say what
  *another* 200M tokens would add, only that each stretch here added less than the one before.
- **No forgetting:** Wikipedia stayed within 1.902–1.908 the whole run, and slightly improved.
  Training would have stopped itself if either loss rose 3 checks in a row; it never came close:
  Wikipedia rose three times by at most 0.002 (500 → 750, 1500 → 1750, 2750 → 3000), never twice in a row.

**What this does and doesn't show.**
- **The extra gain is 10-K-specific.** Compared with the 25M run, the 10-K loss fell another 0.0925,
  Wikipedia only 0.0118. This fits the untested idea from the first run (the model getting back into
  practice at plain text): Wikipedia was already at 1.908 by the first check, better than the whole
  25M run's 1.914, then hardly moved, while the 10-K loss kept falling. As before, "10-K-specific"
  includes this collection's style, not only finance knowledge.
- **Rank 64 vs. rank 16 isn't separated.** The first check (update 250, ~16M tokens) already beat
  the 25M run's final 10-K loss (1.3066 vs. 1.3170), which hints that the bigger add-on learns faster.
  But the two runs also differ in learning-rate schedule, so that's a hint, not a measurement.
- **The assistant wasn't tested.** Whether this helps on the 200 questions (v2: 75.0%) needs the
  fine-tune redone on top, which wasn't run.

**Still a chat model.** The same five questions: "What is the capital of Australia?" gets the
identical answer (Canberra); the other four are reworded but still right: growth still 25%, the
haiku still 5-7-5, the two longer answers still clear and structured.

**Speed and cost.** Two L40S machines (one GPU each, working in step, "DDP"), **10 h 48 min** by the
trainer's clock for 199,999,488 tokens (48,828 pages): **5,140 tokens/s**, peak GPU memory
**33.0 GiB** of 44.4, no interruptions. That time includes the 13 checks (871 s in all); without them,
~5,260 tokens/s. About **$21** for the run (two interruptible machines at ~$0.95/h each), plus
the setup and the short test runs before it. It's 1.9× the first run's 2,685 tokens/s, but that's not a clean
two-GPU scaling number: the add-on is 4× bigger here (the fine-tune's measured figure is 1.86×,
`../qlora_financial/profiling/report.md`).

**Which add-on was scored.** The add-on saved at the end (update 3052) is the one in `after.json`.
The trainer also kept the save point with the best 10-K check (update 2750); the two differ by
0.00003 in 10-K loss (1.22447 vs. 1.22450), so the choice doesn't matter here.

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
   `pretrain.py` now uses the trainer's own `train_runtime` and counts pages actually read; the
   200M run confirmed it: 199,999,488 tokens, exactly its 48,828 pages (the last update held 12).
6. **About 28 idle minutes.** Training ended at 00:20 UTC, but my background check only reported it
   at 00:48 (the cause wasn't found), so the node sat idle and billed for ~28 minutes before the
   "after" measurement and shutdown.

From the 200M run:

7. **Two machines: DDP, not FSDP.** FSDP splits the model itself across GPUs and passes its pieces
   back and forth every layer; between two machines on an ordinary ~1.5 GB/s network that was
   estimated slower than one GPU. DDP keeps a full model on each GPU and only shares the add-on's
   updates, so it was used instead.
8. **Test the restart before the long run.** Interruptible machines can be taken back at any time,
   so a short run was killed on purpose at update 12 and restarted with `--resume`: it picked up from the
   update-10 save point and continued with matching losses (1.500/1.491 vs. 1.501/1.490 uninterrupted). That test also found
   a wrong speed in the summary after a restart, fixed before the real run.
9. **A fresh machine can lack build tools.** The first two-GPU test crashed on the second machine:
   Liger's kernels are compiled on first use and needed Python's header files (`python3.12-dev`),
   missing from that machine's system image.
10. **The stop rule has no minimum size.** "Rose 3 checks in a row" counts a 0.0002 rise like a
    big one; Wikipedia's loss wobbled by up to 0.002, so three tiny rises in a row could have
    stopped a healthy run. A threshold would be safer; not changed, since it didn't trigger.
11. **The cloud login can expire overnight.** The run ended at 08:25 UTC, but deleting the second
    machine needed a browser login, so it waited for the morning. Both machines were shut down from
    inside meanwhile, which the cloud counted as stopped.

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
# the 200M follow-up on two machines, one GPU each (~10 h 48 min): the same command on both, with
# --node_rank 0 on the first and 1 on the second; add --resume to continue after an interruption
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 torchrun --nnodes 2 --nproc_per_node 1 --node_rank 0 \
    --master_addr <first machine's internal IP> --master_port 29500 -m training.cpt_financial.pretrain \
    --lora-rank 64 --tokens 200000000 --warmup 100 --run-name run_200m_r64
python3 -m training.cpt_financial.evaluate --adapter training/cpt_financial/outputs/run_200m_r64/adapter --out after.json
# tests (made-up token files, no GPU): pip install pytest first
python3 -m pytest training/cpt_financial/tests
```

## Files

- `pretrain.py`: the pages (`PageDataset`), the model with bf16 LoRA, the training settings.
- `evaluate.py`: check-pile and WikiText loss, and the five fixed chat questions (`--model` for
  a merged model).
- `merge.py`: folds the reading add-on into the model's weights, for the fine-tune on top.
- `tests/test_pretrain.py`: the pages have the right length and content, the same seed gives the
  same pages, pages only come from the files given, and the page count of a partial last update;
  for the follow-up, the stop rule, the add-on size setting, and that check pages come only from
  the held-out files.
- `results/`: `before.json`, `after.json` (losses and the five answers), `summary.json` (the run's
  speed, memory and every logged loss; note its token overcount, explained above),
  `merged_check.json` (the merged model's losses and answers), `finetune_comparison.json` (v2 vs.
  the new fine-tune and the two starting models: accuracy per dataset, sign tests, and right/wrong
  for each of the 200 questions).
- `results/run_200m_r64/`: the follow-up's `after.json`, `summary.json` and `eval_history.json`
  (the 13 checks with the learning rate at each).
- Not in git: the follow-up's add-on (321 MB) and save points, on the Nebius node only; the first run's add-on (81 MB), the merged model (15 GB, `/home/ubuntu/cpt_merged`), the
  new fine-tune (`training/qlora_financial/outputs/v2_cpt/`, with its full answers) and the run logs
  are on the Nebius node; the logs and both runs' full answers are also in the Mac's git-ignored
  `outputs/results/`.

## Next steps

At 25M tokens, reading 10-Ks didn't make the fine-tuned assistant better. The 200M follow-up
improved 10-K perplexity further, but its effect on the assistant is untested. Options from here:

1. **Fine-tune on top of the 200M model** and score the 200 questions against v2's 75.0%, exactly
   as was done for the 25M run (~3 h on one GPU, ~$3). It's the only way to learn whether the
   extra 10-K gain reaches the answers. A difference of a few questions would again be within noise.
2. **Stop here:** the first run's fine-tune showed no gain, and the arithmetic slips that cost
   answers are a different skill from predicting report text.
3. **Other directions, if revisited:** a task closer to reading (answering from long 10-K passages);
   the control run on general text, only if the perplexity gain itself becomes the question.
