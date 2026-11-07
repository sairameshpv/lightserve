# infra: the platform as code

One `terraform apply` builds a GPU Kubernetes cluster on Nebius with everything the project
runs; `terraform destroy` removes it. The single-VM setup in `../terraform/` stays as it was.

```
infra/
  terraform/        the cluster (one L40S node runs everything) and every Helm release
  helm/
    releases.yaml   the one list of charts, versions and settings files (Terraform and the
                    kind rehearsal both install from it)
    values/         each chart's settings
    lightserve/     lightserve's own chart (fake engine or GPU engine)
    platform/       the repo's monitoring files (alerts, SLOs, dashboards), MailHog, MLflow's disk
  docker/           lightserve's image (fake and gpu targets); CI pushes it to ghcr.io
  kind/rehearse.sh  the same charts on a local kind cluster: free, no GPU
  smoke_test.sh     PASS/FAIL per check, against whatever kubectl points at
```

## What runs on the cluster

| Release | What it's for |
|---|---|
| gpu-operator | Kubernetes sees the GPU (driver already in Nebius's node image: `drivers_preset = cuda12.8`); GPU metrics (DCGM) |
| monitoring | Prometheus, Alertmanager (email to MailHog), Grafana |
| platform | `monitoring/alerts.yml` + `slo.yml` as Prometheus rules, both dashboards, MailHog, MLflow's disk |
| tempo, loki, alloy | traces (lightserve sends OTLP) and logs (Alloy ships every pod's log) |
| mlflow | the registry, its database and artifacts on a disk in the cluster (no managed Postgres or bucket) |
| airflow | the retraining DAG, straight from GitHub (gitSync): **parsed, and its whole-loop test run in the Airflow image; not runnable as deployed** (the cluster's MLflow has no champion, no traffic window or vLLM address reaches the pod, and `MLOPS_RUNS` is a pod's `/tmp`). The real run was step 9, on the VM |
| chaos-mesh | the failure experiments (Phase 4) |
| lightserve | the server on the L40S: Llama-3-8B's shape with random weights |

## One node, not two

The plan had a small CPU node for the platform and an L40S node for lightserve. The first real
apply (1 Nov) never got the CPU node: Nebius caps regular (non-GPU) vCPUs at **3** in eu-north1,
the only region with L40S GPUs, and declined a raise (3 Nov). GPU machines count against the GPU
quota instead, so **the one L40S node (24 vCPUs, 96 GB) runs everything**, with no taint fencing
it off (every pod, Kubernetes's own included, must land there).

- **It fits easily:** all pods ask for ~5.6 vCPUs and ~21 GB (lightserve alone 4 vCPUs, 16 GB);
  even every container at its memory limit is ~53 GB. (Added up from `helm template` of every
  release with the real settings.)
- **Same GPU as the VM sessions**, one machine instead of two (slightly cheaper).
- **Chaos changes:** "GPU node drain" becomes **evict lightserve only and cordon the node**:
  draining the only node would also evict the monitoring that's meant to see it. The noisy
  neighbour gets more realistic: the platform's pods really share lightserve's node.
- **On-demand only:** with `preemptible = true`, one reclaim would take the whole platform down.

First attempt's timings (cluster and GPU node built, CPU node stuck, all destroyed): cluster
5m10s, GPU node group 9m16s, destroy 6m11s. So a session is ~25 min of build + teardown plus the
installs (not yet measured), and the tests are kept to ≤ 10 min.

## Two changes from the plan

- **lightserve runs random weights, not the real Llama-3-8B.** It has no weight loader (cut from
  scope on 29 Oct), so the switch is the model's *shape*: 4 layers (toy, fast start) or 32 (the
  full 8B shape, `lightserve_n_layers`). Still a real GPU workload for chaos and tuning.
- **The image lives on ghcr.io, not a Nebius registry.** A registry made by this Terraform would be
  destroyed with each cluster session, so CI would have nowhere to push in between.
  `.github/workflows/image.yml` pushes `ghcr.io/<owner>/lightserve:{fake,gpu}-<sha>` and
  `airflow-mlops` (Airflow 3.3.2 + the pipeline's packages) with the run's own token (no stored
  secret). The packages are public, like the repo (checked: an anonymous pull gets HTTP 200), so
  the cluster needs no pull token.

## Rehearse on the Mac (free)

```
infra/kind/rehearse.sh up                 # kind cluster + the fake lightserve image
infra/kind/rehearse.sh install <release>  # or "all" (GPU operator skipped)
infra/smoke_test.sh --no-gpu [checks...]
infra/kind/rehearse.sh down
```

Docker Desktop here has 3.8 GB and 8 CPUs: **not enough for everything at once** (with monitoring
and the log/trace stack together the node thrashed: Kubernetes's API took minutes per call). So
the rehearsal went in batches, each installed, checked and removed.

**What the rehearsal found and fixed** (each would have broken the real apply too):

| Found | Fix |
|---|---|
| Alertmanager never started: "undefined receiver null" | the chart's own Watchdog route needs a `null` receiver next to ours |
| `LightserveDown` can't fire on Kubernetes: a killed pod's target *vanishes*, `up` never reads 0 | `or absent(up{job="lightserve"})`, promtool-tested; on kind: pod gone 01:22:36 → firing 44 s later → email 7 s after |
| a healthy lightserve restarted under load: health check limit 1 s | liveness 5 s, readiness 2 s |
| Prometheus at ~90% CPU: every target scraped, every rule evaluated, every 5 s | 15 s default; lightserve's own ServiceMonitor stays 5 s |
| `helm --wait` hung on MLflow's disk (it binds only when MLflow first uses it) | the platform release doesn't wait |
| rules/ServiceMonitors silently skipped if installed before monitoring | created only when monitoring's definitions exist; Terraform installs in two stages |
| MLflow OOMKilled at 1 GB, then 1.5 GB | MLflow 3.16's background job system (7 queue workers + a runner, ~220 MB each, ~2 GB): off (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`): 187 MB |
| MLflow refused other pods: accepts only localhost Host headers by default | `--allowed-hosts` with the cluster names |
| Airflow never finished installing: its database jobs are Helm hooks (run *after* install) while `--wait` waits for pods that wait for the database | the jobs as ordinary Jobs (`useHelmHooks: false`) |
| Airflow containers took minutes to start: each one pip-installed MLflow + Evidently at every start | an Airflow image with them built in (`infra/docker/airflow.Dockerfile`, pushed by CI) |
| the chart's user-creation job failed | Airflow 3 has its own simple login: job off |
| the whole-loop test OOM-killed inside the scheduler | run in a pod of its own (same image) |
| the GPU image didn't build: torch 2.13.0 has no CUDA 12.8 build | torch 2.11.0, the newest cu128 build (as training used on the L40S) |

**Checked on kind, every smoke check but the GPU one PASS:** lightserve (health + completion);
Prometheus scraping it with the repo's rule groups; both dashboards in Grafana; an alert → email in
MailHog; a trace in Tempo and a request log line (with its trace id) in Loki; MLflow logging a run
+ artifact and registering a model from another pod (all still there after deleting MLflow's pod);
Airflow listing the DAG from GitHub with no import errors and **all 4 whole-loop scenarios passing
in the Airflow image**; Chaos Mesh killing the lightserve pod (`infra/chaos/pod-kill.yaml`), a new
pod Ready 5 s later. CI built and pushed all three images.

## The real apply (session Cluster-1, < 1 h, needs a go-ahead)

```
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars            # parent_id, subnet_id
export TF_VAR_nebius_token=$(nebius iam get-access-token) TF_VAR_grafana_admin_password=...
terraform plan                                           # 18 to add
time terraform apply                                     # record how long
$(terraform output -raw kubeconfig_command)
../smoke_test.sh                                         # all PASS
time terraform destroy                                   # record how long
```

Then check **nothing is left behind** (a forgotten disk or cluster keeps billing):
```
nebius mk8s cluster list --parent-id <project>          # no clusters
nebius compute disk list --parent-id <project>          # only the VMs' own boot disks
```

## What the real cluster showed (Cluster-1, 3 Nov, one node, 40.5 min, all destroyed)

| Part | Time |
|---|---|
| apply: cluster 4m42s, GPU node 1m55s (9m16s on the first attempt), the installs ~6.5 min | **13.3 min** |
| smoke test | 4.8 min |
| destroy: the installs ~2.3 min, GPU node 6m42s (1m11s on the first attempt), cluster 4m24s | **13.7 min** |

- **7 of 10 checks passed at once:** the GPU (with the GPU operator's own container toolkit
  on, which works with Nebius's driver), Prometheus with the repo's rules, both dashboards, an
  alert as email, MLflow (its disk bound by the default storage class), **Airflow's whole-loop
  test, 4 of 4**, Chaos Mesh. The CLI token worked as the providers' login.
- **lightserve's GPU image had never run on a GPU, and its engine died on the first request:**
  Triton compiles a small C helper and caches kernels in `~/.triton` on first use; the image had
  no C compiler and the user no home folder. Proven in the session (run as root with `gcc`
  installed: first request 13.5 s, compiling the kernels; then 0.06 s; lightserve, traces and
  logs PASS: all 10) and fixed in the image (c8ff342).
- **Worse: the dead engine was silent:** `/health` kept saying ok and requests waited forever,
  so nothing restarted it and no alert fired. Fixed: `/health` now answers 503 "engine dead" and
  every request ends as aborted (0738dd7).
- **The smoke test warms lightserve up** with one request before the checks (the first one
  compiles the GPU kernels, ~14 s) and reports how long it took.
- **Teardown stays as it is.** Skipping the uninstalls would save ~2.3 min, but they're what
  deletes the cloud disks behind MLflow's and Loki's storage; the slow parts (node and cluster,
  ~11 min) are Nebius's own. Checked after the session: no disks or clusters left.
- One network-controller pod (`cilium-operator`'s second copy) stays Pending: it wants a second
  node for its backup copy. Harmless on one node.

## Chaos experiments: break it on purpose, check it heals (8j)

**What it's for:** every break should be **caught** (an alert, or a number fixed in advance) and
**heal on its own** (back to normal with nobody touching it, no request left hanging). The one
exception is written into its rule: a drain on a one-node cluster needs a person to let the node
back. Like a fire drill, it only proves something if someone is in the building: a load pod sends
steady requests the whole time and records every one's fate (`infra/chaos/load.py`: ok, aborted =
the server's own clean 60 s timeout, error, or hung = no answer within 90 s).

**How one experiment runs** (`infra/chaos/run_chaos.py <name>|all --out runs/chaos/<where>`, from a
laptop whose `kubectl` points at the cluster): today's code is shipped to the pods in a ConfigMap
(so it tests uncommitted code too); steady load (4 users); 30 s normal, 90 s fault, 60 s recovery;
the fault is switched off **at** the end of its phase (every fault file also has a 90 s timer of
its own, in case the runner dies); then the records, Prometheus (read through the Kubernetes API)
and Kubernetes give the facts, and a fixed rule gives PASS/FAIL with its reasons. Saved per
experiment: `commands.log`, `records.jsonl`, `facts.json`, `verdict.json`. `--dry-run` prints
every command and changes nothing.

### The six experiments, predictions and pass rules (written before running)

| # | Fault | Prediction | Caught if | Healed if |
|---|---|---|---|---|
| 1 | Kill lightserve's pod (`pod-kill.yaml`) | a few refused requests; a new pod in seconds, maybe before the down alert's 30 s wait | the load saw failures, or LightserveDown pending/firing | new pod Ready, recovery 100% ok, none hung, in-flight 0 |
| 2 | +100 ms delay, 5% loss (`network-delay.yaml`) | first token ~100 ms later | first-token p50 up >= 80 ms | p50 within 20% of normal, none hung |
| 3 | Flood: 300 extra users (above lightserve's batch of 256) | the server fills up; slower; nothing lost | **the server hit its 256 limit** (see "rule changed" below) | no restart or out-of-memory kill, every request served or timed out cleanly, queue 0, latency within 20% |
| 4 | Noisy neighbour: 20 CPU-burning workers next to lightserve (`noisy-neighbour.yaml`) | GPU: the slowest requests up tens of %; kind's fake engine: meaningless | p99 latency up >= 10% | p99 within 20% of normal |
| 5 | Drain: evict lightserve, fence off the node | LightserveDown fires (the pod has nowhere to go) | LightserveDown firing within 90 s | **human step** logged (uncordon), Ready again within 3 min (GPU 5), alert clears, none hung |
| 6 | MLflow down 90 s during a registration (`mlflow-down.yaml`, `register_probe.py`) | the first try fails or waits; the retry finishes | the first try failed or waited > 10 s | exactly one new version, champion label on it, all its facts (a test model, never finance-qa) |

**Found by reading, before any experiment ran:** if MLflow died between "version created" and
"champion label moved", the pipeline's retry said "already registered" and stopped: a
half-registered model. Fixed (`registry.finish_champion`, called by the DAG's `register_once`),
with a test that cuts the registration off at exactly that point. Chaos can't hit that 1-second
window reliably, so the test, not experiment 6, is the proof.

### Rehearsal on kind (5-6 Nov, fake engine, Mac Docker: 8 CPUs, 4 GB)

| # | Result | What it showed |
|---|---|---|
| 1 Pod kill | **PASS** | 176/176 ok before; 7 refused while the new pod came up; recovery 281/281 ok. The down alert never went pending: the pod was back before the 30 s wait, as predicted possible. "Caught" here = the load saw failures, i.e. the break itself, **not monitoring** (see Cluster-2) |
| 2 Network | **PASS**, prediction wrong | first token 39 -> 242 ms (+203, predicted ~+100), back to 39. Probably each request opens a new connection, paying the delay on the handshake too (not checked). The plan's "+~100 ms, not more" ceiling isn't in the written rule, which has no upper bound |
| 3 Flood | **FAIL, untrustworthy on the Mac** | run 1 (300 users): 303 in flight, batch at 256, p50 1.5 s, all ok, but the flood ran ~10 s into recovery (fixed: stopped at fault end). Run 2, same settings: 55 in flight, p50 17 s, 6 refused. The bottleneck moved outside lightserve: kind, monitoring and 300 load threads share the Mac's Docker machine |
| 4 Noisy neighbour | **FAIL (not caught), as predicted for kind** | first run "PASS" (p99 1.53 -> 3.52 s) was a **false positive**: its normal phase was still slow from the flood just before. Rerun with a settle and the baseline check: 795 -> 708 ms, no effect, although the burn was real (the kind machine's CPU 30% -> 829%, all 8 cores). The fake engine mostly sleeps, so a busy CPU barely touches it: only the GPU cluster can answer this one |
| 5 Drain | **FAIL: a real finding, open** | caught every time (firing after 40-55 s); human step logged; Ready again in seconds; **but requests hung 90 s**: 1, then 4, then 4 |
| 6 MLflow down | **PASS** (2nd try) | 1st try: **every registration failed, fault or not** (below). After the fix: the first try waited 127.5 s (MLflow's client sat out the 90 s outage), then exactly one version, champion, all facts. The pipeline's own retry wasn't needed, so the "cut off before the label" case rests on its test |

**Bugs the rehearsal found in the runner, all fixed with a test first:** the noisy-neighbour file
lacked the labels the runner selects by (crash); results printed only at the end (a crash hid
them); the flood and the neighbour's burn weren't stopped at the end of their phase; a pod that
failed was waited on as if it might still succeed (10 min), and its log was thrown away with it.

**A real bug experiment 6 found, not about chaos:** registering a model against an MLflow
**server** (the cluster's) always failed: "INVALID_PARAMETER_VALUE: Invalid model version source",
because the registry gave a plain folder as the version's source, which only a local database file
accepts. Every test used a local database file; the cluster's smoke test registered differently. So
the cluster's own pipeline could never have registered a model. Fixed: the adapter is uploaded with
the registration run, the version points at the upload, the folder stays a fact (`adapter_path`,
read by the pipeline; old versions fall back to `source`), and against a server MLflow keeps the
files on its own disk, not the client pod's `/tmp`. A test now starts a real MLflow server and
registers against it (it failed with the same error before the fix).

**Rule changed after seeing results (flagged):** the flood's first "caught" rule was "the waiting
queue > 0". On kind the real queue lasted ~50 ms (a slot frees every 20 ms), and Prometheus reads
that gauge every 5 s, so it read 0 although the server was full. Now: "running reached the 256
limit". That gauge is also read every 5 s; it works because a sustained flood keeps the server
full. The size was changed too: the plan's 10x (40 users) never filled a batch of 256.

**The drain hang (open finding).** Timeline of the second run: drain at +30 s; until +39 s the
shutting-down pod refused new connections cleanly (thousands of fast errors); **at +38.8 s all four
users' next requests started and hung**, 0.3 s before the old pod and its network address were
removed. Best explanation, not proven (the address-list recorder logged nothing): for a moment new
connections were still sent to an address that no longer existed, which never answers, not even
"refused", so the client waits its full limit (90 s here, ~2 min with Linux defaults). Any real
client of lightserve without a short connect limit would hang the same way during a drain, restart
or upgrade. Tried: a 10 s "keep answering before shutting down" pause (`preStop`): run 1 still had
4 hung, run 2's drain itself timed out, so it was **taken out again**. Kind's network layer isn't
Nebius's (Cilium), so the race may differ there: **check it first on the real cluster.**

**Two gaps the rehearsal showed, fixed with a test first:** a crashed run left its load pod behind
(now every experiment, finished or crashed, deletes the runner's own pods and every fault object;
never lightserve or MLflow); experiments ran back to back, so one slowed the next one's baseline
(now a 30 s settle before each, `--settle-s`; a run refuses to start while `chaos-*` pods from an
earlier one are still there; and a normal phase more than 1.5x slower than the first experiment's
fails as "BASELINE SLOW" instead of being judged). Checked on kind: a planted leftover pod stopped the
run; network + neighbour left only Chaos Mesh's own pods; network delay repeated its +205 ms.

### Session Cluster-2 runbook (< 1 h, needs a go-ahead)

apply ~13 min -> smoke test ~5 -> `run_chaos.py drain` first (the open finding), then
`pod-kill network-delay queue-flood mlflow-down noisy-neighbour --out runs/chaos/cluster2` ~20 ->
destroy ~14 = ~53 min. Hard stop at start + 55 min: if behind at minute 38, skip the rest and
destroy. On the GPU use `--ready-limit-s 300`. `finish_champion` and the registry fix are on GitHub (the cluster's
own pipeline reads its code from there; experiment 6 ships its own copy). On kind experiment 6 needs `--register-image airflow-mlops:local`.

### Session Cluster-2 results (2026-11-06, 17:26-18:17 UTC, ~51 min, all destroyed)

**In one line:** four of six experiments ran and passed on the real GPU cluster (pod kill, network,
flood, MLflow down); the drain couldn't be judged (its load recorded nothing for 137 s); the noisy
neighbour was cut for time. Nothing left behind: 18 destroyed, 0 clusters, only the VMs' own boot disks.

**Timings:** apply 961 s (16 min; Cluster-1: 13), shortened smoke check ~1 min (gpu, lightserve,
prometheus, alert, mlflow, chaos: 6/6 PASS; first completion 7.6 s incl. compiling the kernels; the
slow Airflow whole-loop check skipped, nothing here needs it), experiments 17:43-18:04, destroy 788 s.

| # | Result on the GPU cluster | What it showed |
|---|---|---|
| 1 Pod kill | **PASS** by the written rule, but **monitoring didn't notice**: 30,106 fast refusals while the new pod started, recovery all ok, none hung, and **no alert at all** (LightserveDown never went pending: back before its 30 s wait). The rule counts the load's own failures as "caught", which is the break itself. There is no error-rate alert: a crash loop that recovers fast would page nobody (open item) |
| 2 Network | **PASS** | first token 24 -> 223 ms (+199) -> 24 ms. Same as kind (+203, +205): the "+100 ms" prediction was wrong everywhere (probably: each request opens a new connection and pays the delay on the handshake too; not checked) |
| 3 Flood | **PASS** | running reached the 256 limit; no restart or out-of-memory kill; every request ended cleanly; queue back to 0. The first trustworthy flood result (the Mac's wasn't) |
| 4 Noisy neighbour | **not run** | cut at the minute-38 rule; still only kind's "no effect on the fake engine" |
| 5 Drain | **FAIL: can't judge** | the load pod recorded **no request at all before +137 s**, the moment lightserve came back (10 s after the uncordon), so no normal phase and no traffic during the drain. What could be seen passed: LightserveDown firing after 55 s, human step logged, Ready 10 s after the uncordon, alert cleared, 0 hung. Cause of the gap unknown (the pod went with the cluster). **The kind hang is neither reproduced nor ruled out** |
| 6 MLflow down | **PASS** | the registration sat out the outage; exactly one new version, champion, all facts: **the registry fix works on the real cluster** |

**Next time for the drain:** run it second, not first (it was the session's first experiment, so
also the first load pod on a fresh cluster), and keep the load pod's own log (`kubectl logs`) before
the pod is deleted, to see what it was doing for those 137 s. Raw records: `runs/chaos/cluster2/` on
the Mac (untracked, like the rest of `runs/`).
