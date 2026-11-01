# infra: the platform as code

One `terraform apply` builds a GPU Kubernetes cluster on Nebius with everything the project
runs; `terraform destroy` removes it. The single-VM setup in `../terraform/` stays as it was.

```
infra/
  terraform/        the cluster (1 CPU node + 1 L40S node) and every Helm release
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
| airflow | the retraining DAG, straight from GitHub (gitSync), in dry-run (the real fine-tune stays on the VM) |
| chaos-mesh | the failure experiments (Phase 4) |
| lightserve | the server on the L40S: Llama-3-8B's shape with random weights |

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
terraform plan                                           # 19 to add
time terraform apply                                     # record how long
$(terraform output -raw kubeconfig_command)
../smoke_test.sh                                         # all PASS
time terraform destroy                                   # record how long
```

**Only the real cluster can tell** (the smoke test's first lines check them): whether Nebius's node
image also has the NVIDIA container toolkit (then `toolkit.enabled=false` in
`values/gpu-operator.yaml`); that the default storage class binds MLflow's and Loki's disks; that
the CLI token works as the providers' `exec` login; how long apply and destroy take (over ~35 min
together and the chaos sessions need re-planning).
