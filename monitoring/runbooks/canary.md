# CanaryAccuracyLow / CanaryStale

The canary (`training/qlora_financial/canary.py`) asks the served model the same 50 test
questions every ~5 min and grades them as the registry does. "Normal" is the champion's
own saved score on those 50 questions (0.66 for v3).

## CanaryAccuracyLow

**What it means:** two canary rounds in a row scored more than 10 points below normal. The
server answers, but the answers are wrong: a wrong or broken model is probably being
served. Speed dashboards look fine in this case. A model only slightly worse than normal
won't trigger it: 50 questions can't tell a few points apart.

**Check:**
1. **Which model is served?** `curl -s localhost:8000/v1/models` should list the
   champion's name (e.g. `v3`), and the canary's `--model` should match it.
2. **What does it answer?** The canary's output prints one line per round. Ask one
   question by hand and see whether the answer looks sensible and ends in `Answer: <number>`.
3. **What changed?** A recent deploy, a new adapter, or a changed chat template.

**Do:**
1. **Roll back** to the previous champion (the registry's `champion` label says which).
2. **If "normal" itself is wrong** (a new champion registered, but the canary started
   before it), restart the canary so it reads the new champion's saved answers.

## CanaryStale

**What it means:** no complete canary round (every question answered) for over 15 min, or
Prometheus can't read the canary at all. Either the model server isn't answering (or
fails some questions, e.g. timeouts over 60 s), or the canary isn't running.

**Check:**
1. **Is the canary running?** `curl -s localhost:9101/metrics | grep canary_` should answer.
   If not, read the end of its output for a traceback.
2. **Is the model server answering?** `curl -s localhost:8000/v1/models`. A rising
   `canary_errors_total` means the canary runs but the server doesn't answer.

**Do:**
1. **Restart** whichever is down: the model server, or the canary with
   `python -m training.qlora_financial.canary --url http://localhost:8000 --model v3`
   (from the repo root, in `mlops/.venv`).
2. **Confirm the alert clears:** a `[RESOLVED]` email follows a few minutes after the next
   complete round.
