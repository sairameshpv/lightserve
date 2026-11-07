# lightserve

**lightserve** is a project for learning how large-language-model serving works and what makes it
fast. It has four parts, each with its own README:
1. **The server:** an inference server built from scratch.
2. **Performance experiments:** run on it and on vLLM.
3. **A finance model:** a question-answering model trained and evaluated end to end.
4. **The MLOps work:** watches, retrains and safely deploys that model.

Every result below was measured on real GPUs (Nebius L40S) against a prediction written before
the run, and failures are reported as failures.

## 1. The server

An LLM inference server written from the ground up, so every layer can be read, measured and
changed:

- **GPU kernels** ([`kernels/`](kernels/README.md)): Triton kernels for matrix multiply, fused
  bias + activation, and FlashAttention (forward and backward), each tested against PyTorch. The
  matrix-multiply and attention kernels tune themselves once per size group.
- **The model** ([`model/`](model/README.md)): a Llama-3-8B-shaped transformer that runs on those
  kernels.
- **The engine** ([`engine/`](engine/README.md)): a scheduler with a paged KV cache (memory handed
  out in blocks, like vLLM's). It also handles chunked prefill (long prompts split so they don't
  stall everyone else), prefix caching (shared prompt beginnings computed once) and preemption.
- **The HTTP front end** ([`server/`](server/README.md)): an OpenAI-style `/v1/completions` API
  with streaming, a request queue, timeouts and Prometheus metrics. It also reports a crashed
  engine as unhealthy instead of hanging.

## 2. Performance experiments

Questions about speed, each answered with a benchmark ([`benchmarks/`](benchmarks/README.md)):

| Experiment | Finding |
|---|---|
| [Prefix caching](benchmarks/prefix_caching/README.md) | time to first token **-68%** when requests share a 2,048-token beginning; no gain below ~256 shared tokens |
| [Chunked prefill](benchmarks/chunked_prefill/README.md) | with 128-token chunks, a long prompt arriving slows everyone else's tokens by only **15%** (20.2 to 23.1 ms) |
| [Speculative decoding](benchmarks/speculative_decoding/README.md) | **+22-29%** tokens/s for a single user, **none** at 5 users, on both lightserve and vLLM |
| [Prefill/decode split](benchmarks/pd_disaggregation/README.md) | on one GPU a long prompt stalls everyone's tokens (+42%); on two GPUs the stall disappears (a 1-vs-2-GPU comparison, as the README notes) |
| [KV-cache compression](benchmarks/kv_int8_compression/README.md), [vLLM fp8](benchmarks/vllm_pd_kv_quant/README.md) | lightserve's own int8 version made decoding **slower** (+13%); vLLM's fp8 gave a real **~13%** win past ~7k tokens, so the idea works and the first version didn't |
| [4-bit weights (AWQ)](benchmarks/vllm_awq_weight_quant/README.md) | per-token time **2.8x faster** for one user, fading to **1.17x** under heavy load |
| [Roofline](benchmarks/profiling/roofline/README.md) | where each kernel sits between memory-bound and compute-bound; the LM head is memory-bound at batch 1 |

## 3. The finance model

A Llama-3-8B fine-tuned (QLoRA) to answer financial questions from documents and tables
([`training/`](training/qlora_financial/README.md)):

- **Fine-tuning:** accuracy on numeric test questions went from **31.5% to 74.5%**.
- **A stricter, fairer checker:** the official 1% tolerance hid arithmetic slips; graded fairly, the
  model scores **60.5%**. A calculator that redoes the model's written steps raised it to **73.8%**,
  and a retrain on step-by-step data (v3) to **81.2%**.
- **Training speed:** the GPU was half idle, waiting on Python to launch work. Batching, bf16
  adapters, fused kernels and FlashAttention-2 took training from 1,343 to **2,579 tokens/s**. A full
  retrain took 3 h instead of 4 h 52 with the same accuracy, and two GPUs gave **1.86x**.
- **Reading 10-K reports first** ([continued pretraining](training/cpt_financial/README.md), on a
  [1.75-billion-token corpus](training/corpus_prep/README.md)): perplexity on financial text
  **-22.7%**, but no accuracy gain once fine-tuned on top (74.0% vs 75.0%).
- **Reinforcement learning** (PPO vs DPO vs GRPO): built and tuned; the full comparison is parked.

## 4. MLOps: watch, retrain, deploy safely

The machinery a team would wrap around that model, so it can notice, fix and undo problems on its
own ([`training/`](training/qlora_financial/README.md#retraining-pipeline-from-changed-questions-to-a-new-champion-or-a-rollback),
[`mlops/`](mlops/dags/drift_retrain.py), [`monitoring/`](monitoring/), [`infra/`](infra/README.md)):

- **Watch it:** metrics, dashboards, traces, logs, email alerts, service-level objectives,
  runbooks, and CI with a security scan.
- **Retrain it safely:**
  - a model registry and a promotion gate
  - query-drift detection and a canary
  - an Airflow pipeline: drift detected, retrain from the champion, score, gate, switch, watch,
    keep or roll back

  **Run for real on the GPU:** the pipeline rejected a weaker candidate (80.3% vs 82.0%), and the
  champion stayed in service.
- **The platform as code:** a Kubernetes cluster on Nebius (Terraform + Helm), rehearsed for free on
  the Mac.
- **Fast and tough:**
  - a busy neighbour made requests **~17% slower**; pinning CPU cores won back about two thirds
  - OS tuning made **no measurable difference**
  - in [chaos tests](infra/README.md#chaos-experiments-break-it-on-purpose-check-it-heals-8j), a
    pod kill, network delay, a flood and an MLflow outage all **healed on their own**, but the pod
    kill paged nobody
- **Found only on the real hardware, each fixed with a test:** the registry couldn't register
  against an MLflow server; a crashed engine looked healthy; the canary watch could roll back a good
  model. Open items (an error-rate alert, a drain re-run, the cost check against the bill, and
  others) are listed in the [infra](infra/README.md) and [training](training/qlora_financial/README.md)
  READMEs.
