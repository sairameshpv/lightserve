# AvailabilityFastBurn / AvailabilitySlowBurn

**What it means:** too many requests are ending as `FINISHED_ABORTED`. The target
(SLO) is 99.5% of requests succeeding, so the 30-day error budget is 0.5%.
- **Fast (page):** over 7.2% are failing, in both the last hour and the last 5 min. At
  this rate the whole month's budget is gone in about 2 days.
- **Slow (ticket):** over 3% are failing, in both the last 6 h and the last 30 min.

"Aborted" has two causes: the **server's timeout** (a request took longer than
`--default-timeout-s`), or **the client disconnected** mid-stream. Only the first is
the server's fault.

**Check:**
1. **Dashboard → "Requests ended per second, by status":** when did aborts start, and
   did total traffic change at the same time?
2. **Is the engine overloaded?** Look at:
   - **"Requests in flight / waiting / running":** a growing `waiting` line means the
     engine can't keep up;
   - **"Queue wait"** and **"KV-cache usage"** near 100%, with **"Preemptions"** going
     up: memory is full and requests are being pushed out;

   then timeouts follow.
3. **Loki:** `{job="lightserve", status="FINISHED_ABORTED"} | json`. Requests with
   `tokens` 0 and `ttft_s` null never got a token; that points to queueing. Click the
   `trace_id` of one to see where its time went.
4. **If none of that is high,** the aborts are probably client disconnects. Check the
   client or load-balancer timeouts.

**Do:**
1. **Overloaded:** reduce traffic or add capacity.
2. **Memory-bound:** fewer `--max-num-seqs`, or more `--num-gpu-blocks` if the GPU has
   room.
3. **Client timeouts too short for long answers:** raise them, or cap `max_tokens`.
