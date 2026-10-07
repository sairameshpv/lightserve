# Corpus-Prep: a financial reading library for pretraining

**What this is.** A pipeline that turns six years of SEC 10-K annual reports (the yearly report every
US public company files) into a clean, deduplicated reading library for an AI model, already
translated into the numbers the model reads. It's meant for later **continued pretraining** of
Llama-3-8B: letting the model read lots of real financial writing so it knows the domain better.
This folder only prepares the reading material; training on it is a separate, later project.

**Where the reports come from.** EDGAR-CORPUS on Hugging Face (`eloukas/edgar-corpus`,
Apache-2.0 license): the text sections of 10-K filings from 1993 to 2020, one filing per line. We
use **2015-2020**. Its own `train` split becomes our **study pile**, and its `validate` split our
**check pile**: held-out text that is never trained on, used to measure how well the model reads
unseen reports.

## The result

One full run on a Nebius node (24 cores), 2026-10-07:

| | Study pile (`train`) | Check pile (`val`) |
|---|---|---|
| Reports in | 33,817 | 4,231 |
| **Reports kept** | **29,580 (87.5%)** | **3,420 (80.8%)** |
| **Tokens** | **1,566,194,221** (16 files) | **180,035,305** (2 files) |
| Size | 6.3 GB | 0.7 GB |

In total **1.75 billion tokens, 7.0 GB**. The whole run took **14 minutes**: 93 s to download
~10 GB, 254 s for pass 1 (taking notes), 488 s for pass 2 (translating and saving), with about 21
of the 24 cores busy on average. The output stays on the node, in `/home/ubuntu/corpus`, where a
later training run would use it. Its report card and index card are in this folder:
`full_run_stats.json` and `full_run_meta.json`.

## How it works

**The main challenge: too big to open at once.** The reports are ~10 GB of text, and the pipeline
was built on a Mac with 8 GB of memory. So it works like a **conveyor belt**: read one report,
process it, save the result, move on. It goes over the reports **twice**. Pass 1 only takes notes
on each report (is it good? is it a copy?); between the passes it decides what to keep; pass 2
re-reads the keepers and saves them in the final format.

1. **Download.** Fetch each year's two files (study and check), 8 at a time.
2. **Basic cleanup.** Fix odd characters (special spaces, joined letters like "ﬁ"), squeeze extra
   spaces, and delete lines that are only page numbers or "Table of Contents".
3. **Quality control.** Throw out reports that are mostly junk: under 200 words, odd word lengths,
   mostly numbers, or more than 20% of the text in repeated lines.
4. **Remove exact copies.** Each report gets a short code computed from its text; a repeated code
   means a word-for-word copy, and only the first is kept.
5. **Remove near-copies.** Each report gets a 128-number "fingerprint" (MinHash); reports whose
   fingerprints say they're 80% or more the same are near-copies, and only the first is kept.
   A bucket system (LSH) means each report is compared only with likely matches, not with all others.
6. **Prevent cheating ("decontamination").** Remove any report that shares a *rare* sentence with
   the fine-tuning test material, so a model trained on this library can't have read the test's
   answers.
7. **Translate for the AI ("tokenize").** Turn each kept report into numbers with Llama-3's own
   dictionary, add an end-of-document marker, and append it to files of ~100 million tokens each.
8. **Report card.** Save how many reports each step removed, the tokens per file, and how long each
   pass took.

**Using many cores: a head chef and helpers.** Most of the work on one report doesn't depend on any
other report, so **helper processes** do it for many reports at once: cleaning, the quality rules,
fingerprints, finding test sentences, and translating. But "is this a copy?" means "a copy of
something seen *earlier*", so those decisions stay with one **head chef** (the main process), who
reads the helpers' notes **in the original order**. The output is therefore byte-for-byte the same
with 1 helper or 24, on the Mac or on the node (checked on a 200 + 200 report sample: 1 and 8
helpers on the Mac, 24 on the node, identical file fingerprints).

## What each step removed, and why

| Reason | Study pile | Check pile |
|---|---|---|
| Near-copy | 2,043 (6.0%) | 419 (9.9%) |
| Exact copy | 842 (2.5%) | 205 (4.8%) |
| Too much repeated text | 698 (2.1%) | 87 (2.1%) |
| Shares a rare test sentence | 544 (1.6%) | 85 (2.0%) |
| Under 200 words | 110 (0.3%) | 15 (0.4%) |
| **Kept** | **29,580 (87.5%)** | **3,420 (80.8%)** |

The other two quality rules (odd word length, mostly numbers) removed nothing: they are a safety
net. What the removals turned out to be, from reading flagged reports in **samples** (the full
run's 2,043 near-copies and 842 exact copies were not inspected one by one):

- **Near-copies: in a 2018 + 2019 check-pile sample, 24 of 25 were template filings from
  *different* companies.** Two investment trusts' 4,143-word reports differ in 4 places (a
  registration number and a date); two sister funds (Invesco DB Agriculture Fund vs. Invesco DB
  Commodity Index Tracking Fund) have ~33,000-word reports differing in 367 places, starting with
  the fund name and founding date. That sample had few companies in both years (68), and 25 of
  those same-company pairs measured 66-86% similar, so most stay under the 80% line. How the full
  study pile, which has the same companies year after year, splits between the two kinds is not
  measured.
- **Exact copies: all 7 in the 2019 check pile were the same text filed under a different company
  number** in the same year, the usual sign of a parent company and its subsidiaries filing one
  combined report (the companies' names were not looked up).
- **"Too much repeated text"** catches reports with whole paragraphs pasted several times (one
  sample report repeated 1,384-character paragraphs 3-4 times).

**Why the check pile lost more:** its reports were compared against the study pile too (the study
pile is processed first). A check report that copies a study report is removed, so the held-out
pile doesn't overlap what the model studies.

## Lessons along the way

Each was found by running a step on real reports and reading what it did, before the full run.

1. **Page headers fooled the repetition rule.** The first version counted repeated *lines* and
   rejected 11 of 200 sample reports; the ones checked looked genuine (4 had 10,000-72,000 words,
   96-97% real words; 2 were read closely). 10-Ks print short page
   headers ("Notes to Consolidated Financial Statements", the company name) and "•" bullets on
   their own lines throughout. Counting repeated *characters* instead (the second version of the
   same rule in DeepMind's Gopher paper) left 1 rejection, a report with 29% of its text repeated.
2. **Boilerplate swamped cheating prevention.** "Remove any report sharing a test sentence" flagged
   1,042 of 1,359 sample reports (77%). The most shared "sentence" was the title of Item 5, found in
   971 of them; next came standard accounting language. Text copied from one specific report is
   *rare*, so the rule became: only sentences found in reports from at most 2 companies count. That
   flagged 44 (3.2%), including company-specific passages (a merger's debt financing, a bank's
   credit-loss example). It counts companies, not reports, so one company repeating its own sentence
   over the years still counts as one leak.
3. **Rarity needs scale.** Contamination was 3.2% on the 1,359-report sample and 7% (28 of 400) on
   the 200 + 200 sample, but 1.7% in the full run (629 of 38,048): with more reports, more shared
   sentences reveal themselves as boilerplate.
4. **The default "end" marker is the wrong one.** The Instruct model's tokenizer ends text with
   `<|eot_id|>`, which means "end of a chat turn". Between documents the right marker is
   `<|end_of_text|>` (id 128001), so the writer names it explicitly.
5. **Faster, with identical output.** On the Mac, 8 helpers built the 200 + 200 sample in 19.0 s
   instead of 48.7 s (2.6×, not 8×: the Mac's cores are not all equally fast, and the head chef
   reads and writes alone). The output files were byte-for-byte identical. The ~2 hours for the full
   run on one core is an estimate from the sample, not a measurement.

## How to run it

From the repository root, with `training/.venv` (it needs `huggingface_hub`, `transformers`, `numpy`; the
Llama-3 tokenizer is gated, so log in once with `hf auth login`):

```
# try it: 200 reports per year and pile, into a scratch folder
python3 -m training.corpus_prep.prepare_corpus --years 2019 --max-docs 200 --out-dir /tmp/corpus_try
# the full build (2015-2020); --workers defaults to all cores, --workers 1 = no helpers
python3 -m training.corpus_prep.prepare_corpus --out-dir <output folder>
# read back 300 random tokens as text
python3 -m training.corpus_prep.prepare_corpus --inspect --out-dir <output folder>
# the tests (12, made-up data, no download): pip install pytest first
python3 -m pytest training/corpus_prep/tests
```

On the Nebius node the full build ran as root with the HF token loaded, as in the rest of this
repo: `sudo bash -c "set -a; . /root/hf_token.env; set +a; <venv>/bin/python -m ..."`.

## The output format

Each `train_NNN.bin` / `val_NNN.bin` is one long list of token ids stored as 32-bit unsigned
integers (Llama-3's dictionary has 128,256 entries, too many for 16 bits), with
`<|end_of_text|>` (id 128001) after every report. A report is never split across two files.
`meta.json` lists the files and their token counts (file size ÷ 4 = tokens). A training script
reads fixed-length windows from the long list, the standard layout used by nanoGPT and Megatron:

```
import numpy as np
tokens = np.memmap("train_000.bin", dtype=np.uint32, mode="r")   # nothing loaded until read
i = np.random.randint(0, len(tokens) - 4097)
x, y = tokens[i:i + 4096], tokens[i + 1:i + 4097]                  # input and next-token target
```

## Technical details

- Code: `prepare_corpus.py` (one file); tests: `tests/test_prepare_corpus.py`.
- Cleanup: Unicode NFKC; page-number lines are 1-3 digits only, so a year alone on a line stays.
- Quality: ≥ 200 words; mean word length 3-10; ≥ 60% of words with a letter; ≤ 20% of characters in
  repeated lines.
- Exact copies: 16-byte BLAKE2b hash of the cleaned text.
- Near-copies: MinHash over 5-word shingles, 128 values (CRC32 word hashes, numpy); LSH with 16
  bands × 8 rows; near-copy at estimated Jaccard ≥ 0.8; streaming, the first one seen is kept.
- Decontamination: sentences of ≥ 60 letters/digits after normalization, from
  `qlora_financial/data/val.jsonl` and `test.jsonl`; rare = found in ≤ 2 companies (CIKs).
- Multi-core: `multiprocessing.Pool.imap` (ordered), 4 reports per hand-over in pass 1, 2 in pass 2.
  On macOS, helpers start by re-reading the launching script, so a script that starts the pool must
  be a file (`python -m ...`), not code piped in on standard input.

## Limitations and next steps

- **Small leftovers from cleanup.** Some bullet characters on a line of their own (e.g. "·") get
  through; harmless, but a possible tidy-up.
- **Near-copy chains.** Each report is compared only with reports already *kept*. If A ≈ B and
  B ≈ C but A and C are less than 80% alike, B is removed and C may stay. Full grouping
  (union-find) would remove C too; keep-first was chosen because it works in one streaming pass.
- **The rarity rule's blind spot.** A sentence that really was copied into the test material, but is
  generic enough to also appear in 3+ companies' reports, is treated as boilerplate and ignored.
  Such sentences are generic by definition, so they reveal little about a specific answer.
- **What the near-copies are, at full scale, is not measured** (see above): only samples were read.
- **Next:** a continued-pretraining run of Llama-3-8B on these files (a separate project), checked
  on the held-out check pile, and then on the fine-tuning test to see whether reading 10-Ks helps.
