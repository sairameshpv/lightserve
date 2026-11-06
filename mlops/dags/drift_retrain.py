"""The retraining pipeline: when the questions drift away from the champion's training data,
build fine-tune data like them, fine-tune from the champion, score the candidate, and let the
gate decide. The logic lives in mlops/retrain.py; this file only wires it up.

  drift_sensor -> build_data -> finetune -> load_candidate -> score -> gate -> deploy -> watch -> register_champion
                                                                              \-> reject           \-> rollback

The candidate is loaded into the running vLLM (under its run folder's name) before it's scored;
deploy moves the "current model" pointer to it (the canary follows with --current-file), the old
adapter stays loaded for a switch back; reject unloads it. watch reads the canary from Prometheus:
it holds up -> registered as the new champion, the old adapter unloaded; it doesn't -> pointer
back, the candidate unloaded and recorded as rejected-after-deploy, an alert sent. The last boxes
are safe to retry: each step checks whether it already happened.

Settings (environment variables, read when a task runs):
  MLOPS_WINDOW          the latest query window, one {"prompt": ...} per line (make_traffic.py)
  MLOPS_RUNS            where each run's folder goes (default mlops/runs)
  MLOPS_TRACKING_URI    the registry (default: training/qlora_financial/outputs/mlflow.db); written only by
                        register_champion / rollback
  MLOPS_ARTIFACT_ROOT   where the registry keeps its files (default training/qlora_financial/outputs/mlruns)
  MLOPS_N_LIKE          training rows like the traffic (+ as many replayed; default 800)
  MLOPS_SENSOR_TIMEOUT  seconds the sensor waits for drift before giving up quietly (default 3600)
  MLOPS_DRY_RUN=1       no GPU: finetune writes a stand-in adapter, score copies MLOPS_DRY_ANSWERS
                        (the vLLM calls still happen: on the Mac, against a fake vLLM)
  MLOPS_FINETUNE_CMD    the real fine-tune, with {data} {init} {out} filled in
  MLOPS_VLLM_URL        the serving vLLM (default http://localhost:8000), started with
                        VLLM_ALLOW_RUNTIME_LORA_UPDATING=True; it must see the run folders at the same paths
  MLOPS_CURRENT         the "current model" pointer (default <MLOPS_RUNS>/current.json)
  MLOPS_N_SCORE         questions scored (default 300)
  MLOPS_COST_LEDGER     the GPU-job ledger the real fine-tune and scoring write to (mlops/cost.py;
                        default <MLOPS_RUNS>/gpu_jobs.jsonl); MLOPS_GPUS how many they use (default 1)
  MLOPS_PROMETHEUS_URL  where the canary's numbers are (default http://localhost:9090)
  MLOPS_ALERTMANAGER_URL  where the rollback alert goes (default http://localhost:9093)
  MLOPS_WATCH_MINUTES   how long the new model is watched (default 10)
  MLOPS_WATCH_POLL      seconds between looks; keep it under the canary's round time (default 30)
  MLOPS_WATCH_STALE     seconds without a complete round before rolling back (default 900, as CanaryStale)
Needs the repo root on PYTHONPATH (tasks import mlops.retrain and training.qlora_financial).
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pendulum
from airflow.sdk import PokeReturnValue, dag, get_current_context, task

REPO = Path(__file__).resolve().parents[2]


def setting(name: str, default=None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"set {name}")
    return value


def run_dir() -> Path:
    """mlops/runs/<run id>, made safe as a folder name (run ids hold ':' and '+')."""
    run_id = re.sub(r"[^A-Za-z0-9._-]", "_", get_current_context()["run_id"])
    path = Path(setting("MLOPS_RUNS", str(REPO / "mlops/runs"))) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def tracking_uri() -> str:
    return setting("MLOPS_TRACKING_URI", f"sqlite:///{REPO}/training/qlora_financial/outputs/mlflow.db")


def champion() -> dict:
    """The registry's champion (mlops.retrain.champion_info): adapter, data, saved answers, pool."""
    from mlops.retrain import champion_info
    return champion_info(tracking_uri())


def register_once(champion_alias: bool, tags: dict) -> None:
    """Register this run's candidate. A retry finds it already registered: it only finishes the
    champion label, in case the first try was cut off before moving it (8j)."""
    import mlflow
    from training.qlora_financial.registry import finish_champion, register
    mlflow.set_tracking_uri(tracking_uri())
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    try:
        register(run_dir().name, run_dir() / "adapter", run_dir() / "answers.jsonl", run_dir() / "data",
                 commit or "unknown", setting("MLOPS_ARTIFACT_ROOT", str(REPO / "training/qlora_financial/outputs/mlruns")),
                 champion=champion_alias, tags=tags)
    except ValueError as e:
        if "is already" not in str(e):
            raise
        print(f"already registered: {e}")
        if champion_alias:
            finish_champion(run_dir().name)


def vllm_url() -> str:
    return setting("MLOPS_VLLM_URL", "http://localhost:8000")


def current_file() -> Path:
    return Path(setting("MLOPS_CURRENT", str(Path(setting("MLOPS_RUNS", str(REPO / "mlops/runs"))) / "current.json")))


def gpu_job(kind: str):
    """Record a real GPU step in the cost ledger (mlops/cost.py), under the run's name."""
    from mlops.cost import gpu_job
    ledger = setting("MLOPS_COST_LEDGER", str(Path(setting("MLOPS_RUNS", str(REPO / "mlops/runs"))) / "gpu_jobs.jsonl"))
    return gpu_job(ledger, kind, run_dir().name, int(setting("MLOPS_GPUS", "1")))


def window_prompts() -> list:
    with open(setting("MLOPS_WINDOW")) as f:
        return [json.loads(line)["prompt"] for line in f]


@dag(schedule=None, start_date=pendulum.datetime(2026, 1, 1), catchup=False, tags=["mlops"],
     default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=1)})
def drift_retrain():
    @task.sensor(mode="reschedule", poke_interval=300, timeout=int(os.environ.get("MLOPS_SENSOR_TIMEOUT", 3600)),
                 soft_fail=True)
    def drift_sensor() -> PokeReturnValue:
        """Done when the drift alarm is on; no drift until the timeout = skipped, not failed."""
        from training.qlora_financial.drift import drift
        with open(Path(champion()["pool_data_dir"]) / "train.jsonl") as f:
            reference = [json.loads(line)["messages"][1]["content"] for line in f]
        summary, _ = drift(reference, window_prompts())
        print(f"drift: {summary}")
        if summary["dataset_drift"]:
            (run_dir() / "drift.json").write_text(json.dumps(summary, indent=2) + "\n")
        return PokeReturnValue(is_done=summary["dataset_drift"], xcom_value=summary)

    @task
    def build_data(summary: dict) -> str:
        from mlops.retrain import build_data
        out = run_dir() / "data"
        counts = build_data(champion()["pool_data_dir"], window_prompts(), summary["drifted_columns"], out,
                            n_like=int(setting("MLOPS_N_LIKE", "800")))
        print(f"fine-tune data: {counts}")
        return str(out)

    @task
    def finetune(data: str) -> str:
        out = run_dir() / "adapter"
        init = champion()["adapter"]
        if setting("MLOPS_DRY_RUN", "0") == "1":  # a stand-in: the champion's config, no weights
            out.mkdir(exist_ok=True)
            shutil.copyfile(Path(init) / "adapter_config.json", out / "adapter_config.json")
            (out / "DRY_RUN").write_text("stand-in adapter: no fine-tune ran\n")
        else:
            cmd = setting("MLOPS_FINETUNE_CMD").format(data=data, init=init, out=out)
            with gpu_job("finetune"):
                subprocess.run(shlex.split(cmd), check=True)
        return str(out)

    @task
    def load_candidate(adapter: str) -> str:
        """Into the running vLLM, next to the current model, under the run folder's name."""
        from mlops.retrain import load_adapter
        load_adapter(vllm_url(), run_dir().name, adapter)
        return run_dir().name

    @task
    def score(data: str, model: str) -> str:
        out = run_dir() / "answers.jsonl"
        if setting("MLOPS_DRY_RUN", "0") == "1":
            shutil.copyfile(setting("MLOPS_DRY_ANSWERS"), out)
        else:
            from mlops.retrain import score
            with gpu_job("score"):
                rows = score(vllm_url(), model, data, n=int(setting("MLOPS_N_SCORE", "300")))
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return str(out)

    @task.branch
    def gate(answers: str) -> str:
        from mlops.retrain import gate
        from training.qlora_financial.registry import load_rows
        champ = champion()
        d = gate(load_rows(answers), load_rows(champ["samples"]))
        (run_dir() / "gate.json").write_text(json.dumps({"champion": champ, **d}, indent=2) + "\n")
        print(f"gate: {'PASS' if d['passed'] else 'REJECT'} {d['only_candidate']} vs {d['only_champion']}, "
              f"p = {d['p']:.2g} {d['reasons']}")
        return "deploy" if d["passed"] else "reject"

    @task
    def deploy() -> None:
        """Move the pointer to the candidate (still loaded? else fail: nothing to point at)."""
        from mlops.retrain import loaded, set_current
        name, champ = run_dir().name, champion()
        if name not in loaded(vllm_url()):
            raise RuntimeError(f"{name} is no longer loaded in vLLM; not switching")
        set_current(current_file(), {"model": name, "samples": str(run_dir() / "answers.jsonl")},
                    {"model": champ["name"], "samples": champ["samples"]})
        (run_dir() / "APPROVED").write_text("passed the gate; now the current model\n")
        print(f"deployed: {name} is the current model ({current_file()})")

    @task
    def reject() -> None:
        from mlops.retrain import unload_adapter
        unload_adapter(vllm_url(), run_dir().name)
        reasons = json.loads((run_dir() / "gate.json").read_text())["reasons"]
        print(f"rejected and unloaded, the current model stays: {'; '.join(reasons)}")

    data = build_data(drift_sensor())
    @task.branch
    def watch() -> str:
        """Read the canary every MLOPS_WATCH_POLL s until watch_verdict says pass or rollback."""
        from mlops.retrain import add_round, canary_reading, watch_verdict
        name, started, rounds = run_dir().name, time.time(), []
        window, poll = float(setting("MLOPS_WATCH_MINUTES", "10")) * 60, float(setting("MLOPS_WATCH_POLL", "30"))
        while True:
            r = canary_reading(setting("MLOPS_PROMETHEUS_URL", "http://localhost:9090"), name)
            rounds = add_round(rounds, r)
            verdict, why = watch_verdict(rounds, r["expected"] or 0.0, started, r["switched_at"], time.time(),
                                         window, float(setting("MLOPS_WATCH_STALE", "900")))
            print(f"watch: {verdict} ({why}); last reading {r}")
            if verdict != "wait":
                break
            time.sleep(poll)
        (run_dir() / "watch.json").write_text(json.dumps({"verdict": verdict, "why": why, "rounds": rounds}, indent=2) + "\n")
        return "register_champion" if verdict == "pass" else "rollback"

    @task
    def register_champion() -> None:
        """The new champion (its pool stays the full data), then the old adapter out of vLLM."""
        from mlops.retrain import loaded, read_current, unload_adapter
        before = champion()
        pool = before["pool_data_dir"]
        register_once(champion_alias=True, tags={"pool_data_dir": pool})
        previous = read_current(current_file(), {})["previous"]["model"]
        if previous != run_dir().name and previous in loaded(vllm_url()):
            unload_adapter(vllm_url(), previous)
        print(f"{run_dir().name} is the champion; unloaded {previous}")

    @task
    def rollback() -> None:
        """Pointer back to the model before, the candidate out of vLLM, recorded, alert sent."""
        from mlops.retrain import loaded, read_current, set_current, unload_adapter
        name, champ = run_dir().name, champion()
        why = json.loads((run_dir() / "watch.json").read_text())["why"]
        current = read_current(current_file(), {"model": champ["name"], "samples": champ["samples"]})
        if current["model"] == name:  # a retry after the switch must not switch back to it
            set_current(current_file(), current["previous"], {"model": champ["name"], "samples": champ["samples"]})
        if name in loaded(vllm_url()):
            unload_adapter(vllm_url(), name)
        register_once(champion_alias=False, tags={"pool_data_dir": champ["pool_data_dir"],
                                                  "status": "rejected-after-deploy", "rollback_reason": why})
        alert = [{"labels": {"alertname": "ModelRolledBack", "severity": "page", "model": name},
                  "annotations": {"summary": f"{name} rolled back after deploy: {why}"}}]
        req = urllib.request.Request(f"{setting('MLOPS_ALERTMANAGER_URL', 'http://localhost:9093')}/api/v2/alerts",
                                     data=json.dumps(alert).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).close()
        print(f"rolled back {name}: {why}")

    deployed = deploy()
    gate(score(data, load_candidate(finetune(data)))) >> [deployed, reject()]
    deployed >> watch() >> [register_champion(), rollback()]


drift_retrain()
