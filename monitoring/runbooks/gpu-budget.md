# GpuBudgetExceeded

`mlops/cost.py serve` adds up the GPU-job ledger (`mlops/runs/gpu_jobs.jsonl`): every node or
cluster session writes a start and an end line, and the retraining pipeline's real fine-tune and
scoring runs do too. The month's cost counts **sessions only** (the node's time is what the cloud
bills; a fine-tune on that node is part of it), at the hourly rate given with `--rate`.

**What it means:** this calendar month's sessions (UTC) have cost more than `--budget`, for
5 min. A session still running counts up to now, so the cost keeps rising until it's stopped.

**Check:**
1. **Is a session still running?** The Cost dashboard's job table marks open sessions. Is the
   node or cluster really still up (Nebius console, `terraform state list`)?
2. **Was a session left open in the ledger only?** A session whose node was stopped but never
   recorded as stopped keeps counting. Compare with the cloud console.
3. **Is the rate right?** `gpu_hourly_rate_dollars` should match the price on the bill.

**Do:**
1. **Stop what's running** that doesn't need to be (stop the node, `terraform destroy` a
   cluster), then close its session:
   `mlops/.venv/bin/python -m mlops.cost stop --name <session>`.
2. **If the ledger was wrong** (a session never stopped), close it as above; the cost stops
   rising. The month total is then too high by the time it stayed open; note it.
3. **If the budget is simply too low** for planned work, restart the exporter with a higher
   `--budget`, and say so in the project notes.
