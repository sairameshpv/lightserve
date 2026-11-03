# A/B tool: did setting B make it faster, slower, or can't we tell?

Every Phase 4 change (OS tuning, CPU isolation, chaos) asks the same question, so it's answered
the same way every time: the same load test in turns, **A B B A A B**, every request's own
timings kept, then for each measure B's change vs A in % with a 95% range and a verdict:
**better** / **worse** only when the whole range is on one side of 0, otherwise **not sure**.
A setting is kept only if the deciding measure says "better".

```
python benchmarks/ab/ab_run.py --host http://localhost:8000 --out runs/persistence \
    --a-setup "sudo nvidia-smi -pm 0" --b-setup "sudo nvidia-smi -pm 1" \
    --primary "time/token p50 (ms)"          # named BEFORE the run (default)
```

## How a run goes

1. **Set up the side.** Before each round, that side's shell command (restart the server with a
   flag, change an OS setting...), then wait for `/health`.
2. **Same load, same prompts.** Locust (`benchmarks/locustfile.py`, streaming) with `--seed`:
   every round sends the same prompts in the same order (a random mix of answer lengths moved
   tokens per request ±8% between rounds).
3. **Keep the measured window.** A warm-up is thrown away; only requests that *start* inside the
   window count, and Locust lets in-flight requests finish (cutting them would favour short ones).
4. **One number per round per measure.** Requests in a round share that moment's conditions, so a
   round, not a request, is one sample.
5. **Compare rounds.** B's 3 values vs A's 3 values, a Welch t-interval (the textbook formula
   for two small groups with possibly different noise).
6. **Report.** The decision line, the table, and one line per round (so drift is visible).

## The measures

| Measure | What it is |
|---|---|
| latency p50 / p99 | whole request, start to last token |
| TTFT p50 / p99 | time to first token |
| **time/token p50** / p99 | time per output token after the first: steady whatever the answer's length, so the most sensitive to speed settings. **Default deciding measure.** |
| tokens/s | tokens made *inside* the window per second (each request's tokens spread over its own time) |
| failures | count; a request fails without "200 OK", a token and a finish reason |

**spread** = how far one side's own rounds disagree, (max − min) / mean. A large spread means
the machine changed mid-run (look at the per-round lines).

## What the Mac check showed (fake engine, 3 rounds a side, 30 s rounds, 10 users)

| Check | Deciding measure (time/token p50) | Other measures |
|---|---|---|
| A = B (20 ms per step both) | **not sure**, +0.2% (−2.5 .. +2.9) | 6 not sure, 1 false "worse" (TTFT p99, +2.9%) |
| B 10% slower (22 vs 20 ms) | **worse**, +8.4% (+7.5 .. +9.3) | 6 worse (latency p50/p99, TTFT p50, time/token p99, tokens/s −7.9%), TTFT p99 not sure |

(+8.4%, not +10%: each token also carries ~3 ms of server overhead the step time doesn't change.)

## Limits, found the hard way

- **The first design (a two-level bootstrap) wasn't honest with 3 rounds a side:** in simulation
  it called identical sides different **22%** of the time instead of 5%, and on the Mac it called
  two identical servers "worse". Rounds as the samples + a t-interval: **1.9%**. The price is
  caution: with 3 rounds a side, a clean 10% change is caught about 2 times in 3 when rounds vary
  ±3%; more rounds sharpen it.
- **One deciding measure, named before the run.** Seven measures are seven chances of a fluke:
  about 1 identical comparison in 6 shows a false verdict *somewhere* (the TTFT p99 above). The
  rest of the table is background, not grounds for keeping a setting.
- **p99 needs many requests.** A round's p99 rests on its slowest ~1% (here ~1 request): TTFT
  p99 varied 10-27% between a side's own rounds. Use p99 as the deciding measure only with long
  rounds (thousands of requests each), e.g. the noisy-neighbour test.
- **Tokens/s over "first start to last end" was blurry:** the tail of users finishing one by one
  turned the 10% slowdown into "−3%, not sure". Hence the fixed window.
- **The Mac drifts:** in one run time per token rose 23.3 → 25.6 ms between A's own rounds (heat,
  background work). A B B A A B cancels a steady drift, not a jump; the spread column shows it.
- **One round a side gets no verdict** ("too few rounds"): there's no round-to-round spread to
  judge by.

**Found along the way:** lightserve's streaming dropped tokens slower than 50 ms (a "200 OK"
with only `[DONE]`; 17 of 40 at 8 concurrent): fixed in `server/app.py` (6748876). And Locust's
own latency numbers with `--stream` stop at the response headers (~4 ms vs 2-4 s): the tool uses
its own per-request timings (`--samples-out`).

## Files

- `ab_run.py`: the runner. `--rounds-per-side 3 --warmup-s 30 --round-s 120 --users 10
  --stop-timeout-s 120 --seed 0 --primary ...`; `--out` must be a new folder: `args.json`,
  `round<n>_<side>.jsonl` (every request, warm-up included), `round<n>_<side>_locust.log`,
  `samples.jsonl` (kept requests, tagged), `tokens_per_s.json`, `report.txt`, `report.json`.
- `ab_stats.py`: `welch`, `round_order`, `percentile`, `verdict`, `window_tokens_per_s`,
  `report`, `format_table`. Pure Python, no numpy/scipy.
- Tests: `benchmarks/tests/test_ab_stats.py` (17), run by CI with the other benchmark tests.
- `benchmarks/locustfile.py`: `--samples-out FILE` (needs `--stream`) and `--seed`.
