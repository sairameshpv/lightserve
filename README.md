# lightserve

**lightserve** is an LLM inference server built from scratch to learn how serving works and what
makes it fast: GPU kernels, a scheduler with a paged KV cache, an OpenAI-style HTTP server, and a
long list of performance experiments. Around it sits a **finance question-answering model**
(Llama-3-8B fine-tuned with QLoRA) and, most recently, the **MLOps work** that watches, retrains
and deploys that model safely, summarised below.

Where things are (each folder's README goes into detail):

| Folder | What's in it |
|---|---|
| [`kernels/`](kernels/README.md), [`model/`](model/README.md), [`engine/`](engine/README.md), [`server/`](server/README.md) | the server itself: Triton kernels, the model, the scheduler and KV cache, the HTTP front end |
| [`benchmarks/`](benchmarks/README.md) | performance experiments (speculative decoding, prefill/decode disaggregation, KV-cache and weight quantization, prefix caching, profiling) and the [A/B tool](benchmarks/ab/README.md) |
| [`training/`](training/qlora_financial/README.md) | the finance model: data, QLoRA fine-tuning, evaluation, the model registry, drift, the canary, the retraining pipeline |
| [`mlops/`](mlops/dags/drift_retrain.py) | the retraining pipeline (Airflow) and the GPU cost ledger |
| [`monitoring/`](monitoring/) | Prometheus rules (alerts, SLOs), Grafana dashboards, [runbooks](monitoring/runbooks/) |
| [`infra/`](infra/README.md) | the platform as code (Terraform + Helm), the Mac rehearsal, [OS tuning and CPU isolation](infra/os/README.md), [chaos experiments](infra/README.md#chaos-experiments-break-it-on-purpose-check-it-heals-8j) |

## The MLOps project (Oct-Nov 2026): what was built and what it showed

**What it's for:** a model that's fine on the day it ships can quietly get worse as the questions
people ask change. This project wraps the finance model in the machinery a real team would use to
notice that, retrain it, check the new version is really better, ship it without downtime, and
undo it if it misbehaves, plus the platform to run it all and experiments to make it fast and
tough. The goal throughout was learning, so every result was measured against a prediction
written *before* the run, and failures are reported as failures.

**How it was built and checked:** in small steps, each tested first on the Mac (with fakes where a
GPU was needed), then proven in short GPU sessions on Nebius, each under an hour, each torn down
or stopped afterwards. An independent review at the end checked every published number against
the raw data.

### What was built

1. **Watch it (Phase 1).** lightserve's own metrics, a Grafana dashboard, GPU metrics, request
   traces (Tempo) and logs (Loki), alerts that arrive as email, service-level objectives (SLOs)
   with burn-rate alerts, runbooks, and CI: lint, tests, a smoke test, a security scan.
2. **Retrain it safely (Phase 2).** A model registry with lineage (MLflow); a **promotion gate**
   (a new model must win more questions than it loses, not by luck, with no topic falling more
   than 2 points); **query drift** detection (Evidently); a **canary** asking the served model the
   same 50 questions every few minutes; and a **retraining pipeline** (Airflow): drift detected ->
   build data -> fine-tune from the champion -> load it into the running vLLM -> score -> gate ->
   switch -> watch the canary -> keep it, or roll back and raise an alert. Plus a GPU cost ledger.
3. **The platform as code (Phase 3).** A one-node Kubernetes cluster on Nebius (Terraform) running
   monitoring, MLflow, Airflow, Chaos Mesh and lightserve (Helm), rehearsed for free on the Mac
   (kind) with a smoke test, and images built by CI.
4. **Make it fast and tough (Phase 4).** An A/B tool (alternating rounds, an honest "not sure"); a
   hardware report; CPU isolation against a noisy neighbour; OS tuning one setting at a time; and
   six chaos experiments (kill the pod, slow the network, flood it, a noisy neighbour, drain the
   node, take MLflow down).

### What the real GPU sessions showed

| Session | Finding |
|---|---|
| **Kernel tuning fix** (VM-2) | under real traffic the GPU kernels re-tuned themselves almost every step (~1 token/s); tuning once per size group fixed it: ~167 requests/min instead of ~20 |
| **Noisy neighbour** (VM-2) | a CPU-burning neighbour made the slowest requests **~17% slower**; pinning lightserve to its own cores won back about two thirds (with a smaller neighbour on that side, so not pinning alone) |
| **OS tuning** (VM-3) | GPU clock lock, huge pages, network backlog: **no measurable difference** for any of them; the GPU already runs at full clock under load, so no setting was kept |
| **Chaos on the real cluster** (Cluster-2) | pod kill, network delay, a 300-user flood and MLflow down **all healed on their own**; but the pod kill **paged nobody** (there's no error-rate alert), and the drain couldn't be judged (its load recorded nothing for 137 s) |
| **The retraining pipeline, end to end** (step 9) | ran for real on the GPU: drift detected, a 100-step fine-tune from v3, the candidate loaded into the running vLLM, 300 questions scored in 33 s, and the gate **rejected** it (80.3% vs v3's 82.0%), so v3 stayed in service |

**Bugs that only showed up for real**, each fixed with a test:
- a cut-off registration could leave the model half-registered
- the registry couldn't register against an MLflow *server* at all
- v3's adapter path was a Mac path
- a dead engine thread left `/health` saying "ok" while requests hung forever
- the watch could roll back a good model if the canary switched late

### Still open

Logged with the plan, not hidden:
- check the GPU cost against Nebius's bill
- add an error-rate alert
- re-run the drain experiment
- the noisy neighbour on the real cluster
- the "pinning without a neighbour" comparison
- Kubernetes CPU pinning
- make the cluster's Airflow actually runnable (today it only parses the pipeline)
- two known limits of the gate: it compares the new model's live answers with the champion's
  saved ones, and its per-topic rule is too strict for a 300-question sample

Details, with every number and caveat:
[training README](training/qlora_financial/README.md) (registry, drift, canary, pipeline, step 9),
[infra README](infra/README.md) (cluster, chaos), [infra/os README](infra/os/README.md) (isolation,
tuning), [A/B tool](benchmarks/ab/README.md).
