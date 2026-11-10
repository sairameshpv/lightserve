# RAG demo: the original model vs the fine-tuned one, side by side

A web page where you ask a question about a company's annual report ("What was Apple's revenue
growth in 2019?") and get **two answers side by side**: the original Llama-3-8B-Instruct and the
fine-tuned **v3** (from [the finance model](../training/qlora_financial/README.md)). Both read the
same passages from the companies' 10-K reports, use the same tools, and answer 5 times each with the
same random seeds; the answers are voted on. Only the model differs, so any difference on the page
comes from the fine-tuning.

The page is live only during a GPU session (the models run on vLLM on the node). Every model call is
logged, so the demo's real traffic can be replayed as load through the [A/B tool](../benchmarks/ab/README.md).

## The main challenge

The answer is only as good as its weakest link, like a relay race:
- the right report found (Apple, fiscal 2019, not another company or year)
- the right sentence read
- the arithmetic done exactly
- an answer the runs agree on

So each link has its own tests on the Mac with fakes (no GPU), and the page itself was rehearsed
end to end against a scripted fake model before any GPU time.

## How one question is answered

1. **Find the company and year** (`retrieve.py`). The company by its name ("Apple", "Johnson &
   Johnson"), a name's first word ("Ford" for FORD MOTOR CO, but never the question's own first
   word, so "Total revenue..." is not the fund "Total Return"), or its ticker ("AAPL"). The year is
   the latest one named: "2019 compared to 2018" is the 2019 report, which holds both.
2. **Search only that report** (Qdrant, a vector database). Every passage of every 2019 report was
   turned into 384 numbers that capture its meaning ("embedded"); the question is embedded the same
   way and the closest passages win, filtered to that company and year, so another company's numbers
   can't come back.
3. **Let the model work in rounds** (`agent.py`). The first search is done for it (the same 5
   passages for both models). Each reply is one line: a tool (`search` for 2 more passages, or
   `calculate` for exact arithmetic) or `Answer: -2% | <one short sentence>`. At most 5 rounds and
   3,600 words of passages, and only each earlier reply's tool line stays in the conversation (the
   page still shows the whole reply), which keeps the worst case inside vLLM's 8,192-token limit.
4. **Answer 5 times and vote** (`best_of_n.py`). Seeds 0-4, temperature 0.7, all sent at once so
   vLLM batches them. Answers count as the same number when within 1% or equal once rounded
   ("-2%", "-2.0%" and "-2.04%" are one answer).
5. **Show it** (`app.py`, `static/index.html`). What was understood ("Apple Inc. (AAPL), fiscal
   2019"), a verdict ("Both models give the same answer" / "The models disagree: ..."), each model's
   answer with its count ("3 of 5 runs agree") and sentence, the sources, and every run's steps. For
   the demo questions (`demo_questions.json`), the correct answer from the report and a ✓ or ✗ per
   model.

## What the data can and can't answer

The documents are [EDGAR-CORPUS](https://huggingface.co/datasets/eloukas/edgar-corpus) 2019:
Items 7 (management's discussion) and 8 (financial statements) of **6,693 reports**, split into
**291,759 passages** of ~400 words (1.1 GB in Qdrant; built in 12 minutes on the node's GPU).

- **No tables.** The dataset keeps only the written text, so totals that live in tables are
  missing (Apple's $260.2 billion of 2019 net sales is not there). What companies *write* is there:
  "Total net sales decreased 2% or $5.4 billion during 2019 compared to 2018". Questions about
  changes, rates and reasons work best; exact totals often get "not found".
- **"2019" is the fiscal year that ends in 2019** (checked: Exxon's 2019 report ends December
  2019, Apple's September 2019). One catch: a few retailers name a year that ends in early 2019
  after the year before (Home Depot's "fiscal 2018" ended February 3, 2019), so for them the page's
  "fiscal 2019" is their own fiscal 2018. NVIDIA and Walmart name it by the end, so they match.
- **About 3,160 companies can be asked about**: named in today's SEC ticker list, with at least 5
  passages of management's discussion. Missing: some large companies that the dataset lacks for
  2019 (GE, Citigroup, Verizon, McDonald's, Morgan Stanley); some present with empty sections
  (Intel, Pfizer: the page says so and asks neither model); some that put their discussion in an
  attachment the dataset leaves out (JPMorgan, Chevron, Exxon, IBM: ~1 passage); foreign companies
  (a different form); ~3,200 companies renamed or delisted since 2019 (their text is indexed, but
  today's list has no name for them).

## The demo questions

Five questions, their answers looked up by hand in the indexed text and written down
(`demo_questions.json`, with each quote) before any GPU run. Each tests one thing:

| Question | Answer | Tests |
|---|---|---|
| What was Apple's revenue growth in 2019? | -2% | reading a decrease as a negative number |
| By how much did Netflix's revenue grow in 2019? | 28% | a direct lookup |
| What was NVIDIA's revenue growth in fiscal 2019? | 21% | a fiscal year ending in January (2019) |
| How much did Microsoft's Azure revenue grow in fiscal 2019? | 72% | a trap: the same sentence says 25% for the wider group |
| According to NVIDIA's fiscal 2019 report, what was its automotive revenue in the previous year? | $557 million | the calculator: "$641 million, up 15%" -> 641 / 1.15 (the common slip, 641 x 0.85 = $545 million, is wrong) |

Every answer passage is in the 5 passages both models start with (ranks 1, 1, 5, 3, 2). Dropped: Coca-Cola's operating margin (operating income / revenue): its revenue passage ranks 18th, out of reach
of the follow-up searches, so it would test the search, not the models.

## How it was built and checked

All on the Mac first, test-first, with fakes; 86 tests (`rag/tests`, also in CI):

| Piece | Checked by |
|---|---|
| chunking, names, the index build | 8 tests; a 50-report sample, then the full year on the GPU |
| finding company and year | 17 tests; 20 of 20 large companies right on the real SEC list, 0 of 68 ordinary words mistaken for a company at a question's start |
| search | 3 tests on in-memory Qdrant: the company/year filter holds; on the real index Apple's answer passage ranks 1st for 3 phrasings |
| the agent | a scripted fake model: tools, a garbled reply, too many rounds, bad arithmetic; the worst case measured with the Llama-3 tokenizer, a model rambling ~380 tokens before each of 4 searches: 5,481 tokens + a 512-token reply = 5,993, of 8,192 |
| the vote | rounding, scale words, ties, no-answer runs; a barrier proves the 5 runs are sent at once |
| the page | password (none / wrong / right), notes, verdict, ✓/✗, every demo answer readable as a number; a full rehearsal against a fake vLLM |
| the workload log | thread-safe lines; the chat template matches the real tokenizer's token for token; a 4-second replay through Locust (4,256 requests, all from the log) |

One bug found by the rehearsal: a burst of 10 connections at a fresh server dropped 4 (a small
connection queue). `llm.chat` now retries a dropped connection once (not a server error, which is a
real problem); the same burst then succeeded every time.

## Running it

On the Mac, a rehearsal (no GPU): Qdrant in Docker with the index, a scripted fake on port 8000,
then:
```bash
RAG_PASSWORD=... python -m rag.app --vllm http://127.0.0.1:8000 --rehearsal   # http://localhost:8080
```
The page then says "Rehearsal: scripted fake models, not real answers" at the top.

On the node, during a GPU session: vLLM with the base model and the v3 adapter (step 9's
`vllm_up.sh`), Qdrant on the stored index, then:
```bash
RAG_PASSWORD=... python -m rag.app --device cuda --host 0.0.0.0 --workload runs/rag/workload.jsonl
python -m rag.workload runs/rag/workload.jsonl runs/rag/prompts.jsonl        # the log -> a prompt file
python benchmarks/ab/ab_run.py ... --target vllm --prompts runs/rag/prompts.jsonl   # replay it as load
```

## Later: lightserve

The plan is to serve the same demo through lightserve and compare. Only `llm.py` talks to a model
server, so that means a second version of its one function, plus what lightserve lacks today:
sampling (it is greedy-only, so best-of-N would give one answer 5 times), loading a real checkpoint,
the fine-tuned model (no adapter support: v3's merged copy), and text in and out (it takes token
ids).

## Technical details

- Files: `ingest.py` (build), `retrieve.py`, `llm.py`, `agent.py`, `best_of_n.py`, `app.py`,
  `static/index.html`, `workload.py`, `demo_questions.json`, `tests/`.
- Versions: `rag/requirements.txt` (qdrant-client 1.15.1 for the server image
  `qdrant/qdrant:v1.15.5`; sentence-transformers 6.2.0; FastAPI 0.141.1 / uvicorn 0.52.4 as
  lightserve's server).
- Embeddings: `BAAI/bge-small-en-v1.5`, normalized, cosine distance; questions get bge's search
  prefix, passages don't. HNSW `m=16`, `ef_construct=100`; payload indexes on `cik` and `year`.
- Names: the SEC's `company_tickers.json` (downloaded once, kept in `rag/data/`, not in git).
- vLLM requests: `/v1/chat/completions`, model `meta-llama/Meta-Llama-3-8B-Instruct` or `v3`,
  `temperature=0.7`, `top_p=0.95`, `seed` 0-4, `max_tokens=512`.
- Measured: Llama-3 tokens per word on 300 passages, median 1.28, worst 1.77.
- The page uses HTTP basic auth (`RAG_PASSWORD`; it won't start without one) and shows all model
  text as plain text, never as HTML.
