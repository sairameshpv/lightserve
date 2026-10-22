# LatencyFastBurn / LatencySlowBurn

**What it means:** too many first tokens are arriving later than 0.5 s. The target
(SLO) is 99% within 0.5 s, so the budget is 1%. Fast (page) means over 14.4% are slow,
over 1 h and 5 min; slow (ticket) means over 6%, over 6 h and 30 min. The 0.5 s target
is a placeholder until the GPU baseline is measured.

**Check:** time to first token = time spent waiting in the queue + prefill. Find out
which one grew:
1. **Dashboard → "Queue wait":** if its p99 rose with the time to first token, requests
   are waiting for a slot. Look at "waiting", "KV-cache usage" and "Preemptions".
2. **Dashboard → "Engine step time, prefill vs decode":** if prefill steps got slower,
   prompts got longer or the GPU got slower.
3. **The GPU row:** a dropping SM clock, or a non-zero "clock held back" value, means
   it's throttling (power or temperature). Low "GPU busy" while latency is high means
   the CPU side is the bottleneck: lightserve is launch-bound, see Phase 4.
4. **Tempo:** open a slow request's trace (Explore → Tempo, or a log line's
   `trace_id`). Its `queue` and `prefill` spans show where the time went.

**Do:**
- **Queueing:** reduce load or add capacity, or lower `--max-num-batched-tokens` so one
  long prompt can't hold up everyone else's first token.
- **Long prompts:** cap prompt length at the client.
- **GPU throttling:** check its cooling and power limits on the node (see Phase 4's
  clock settings).
