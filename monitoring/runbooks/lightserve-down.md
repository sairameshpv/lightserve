# LightserveDown

**What it means:** Prometheus couldn't read lightserve's `/metrics` for 30 s straight.
Either the server is down, or Prometheus can't reach it. Clients are probably getting
errors too.

**Check:**
1. **Is it alive?** `curl -s localhost:8001/health` should answer `{"status":"ok"}`. On
   the GPU node, also run `sudo docker ps` to see whether its container is running.
2. **If it's alive,** Prometheus can't reach it. Open http://localhost:9090/targets:
   the `lightserve` target's error says why, for example a wrong port or host.
3. **If it's dead,** read the end of its output (the log file or `docker logs`). Look
   for a Python traceback or CUDA out-of-memory error. The last requests before the
   crash are in Loki: `{job="lightserve"}` in Grafana's Explore.

**Do:**
1. **Restart it:** `python -m server …` with the same options.
2. **If it crashed with out-of-memory,** restart with smaller `--num-gpu-blocks` or
   `--max-num-seqs`.
3. **Confirm the alert clears:** a `[RESOLVED]` email follows within about a minute.
