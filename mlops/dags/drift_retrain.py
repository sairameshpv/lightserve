"""The retraining pipeline: when the questions drift away from the champion's training data,
build fine-tune data like them, fine-tune from the champion, score the candidate, and let the
gate decide. The logic lives in mlops/retrain.py; this file only wires it up.

  drift_sensor -> build_data -> finetune -> score -> gate -> deploy | reject

Settings (environment variables, read when a task runs):
  MLOPS_WINDOW          the latest query window, one {"prompt": ...} per line (make_traffic.py)
  MLOPS_RUNS            where each run's folder goes (default mlops/runs)
  MLOPS_TRACKING_URI    the registry, only read here (default: training/qlora_financial/outputs/mlflow.db)
  MLOPS_N_LIKE          training rows like the traffic (+ as many replayed; default 800)
  MLOPS_SENSOR_TIMEOUT  seconds the sensor waits for drift before giving up quietly (default 3600)
  MLOPS_DRY_RUN=1       no GPU: finetune writes a stand-in adapter, score copies MLOPS_DRY_ANSWERS
  MLOPS_FINETUNE_CMD    the real fine-tune, with {data} {init} {out} filled in
  MLOPS_VLLM_URL        the real scoring server (default http://localhost:8000); the candidate is asked
                        under the name of its run folder (step 8b loads it there)
  MLOPS_N_SCORE         questions scored (default 300)
Needs the repo root on PYTHONPATH (tasks import mlops.retrain and training.qlora_financial).
"""
import json
import os
import re
import shlex
import shutil
import subprocess
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


def champion() -> dict:
    """The registry's champion: its adapter, training data folder and saved test answers."""
    import mlflow
    uri = setting("MLOPS_TRACKING_URI", f"sqlite:///{REPO}/training/qlora_financial/outputs/mlflow.db")
    mv = mlflow.MlflowClient(tracking_uri=uri).get_model_version_by_alias("finance-qa", "champion")
    return {"version": mv.version, "name": mv.tags["version_name"], "adapter": str(REPO / mv.source),
            "data_dir": str(REPO / mv.tags["data_dir"]), "samples": str(REPO / mv.tags["samples"])}


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
        with open(Path(champion()["data_dir"]) / "train.jsonl") as f:
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
        counts = build_data(champion()["data_dir"], window_prompts(), summary["drifted_columns"], out,
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
            subprocess.run(shlex.split(cmd), check=True)
        return str(out)

    @task
    def score(data: str, adapter: str) -> str:
        out = run_dir() / "answers.jsonl"
        if setting("MLOPS_DRY_RUN", "0") == "1":
            shutil.copyfile(setting("MLOPS_DRY_ANSWERS"), out)
        else:
            from mlops.retrain import score
            rows = score(setting("MLOPS_VLLM_URL", "http://localhost:8000"), run_dir().name, data,
                         n=int(setting("MLOPS_N_SCORE", "300")))
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
        """Records the approval; step 8b loads the adapter into vLLM and switches traffic here."""
        (run_dir() / "APPROVED").write_text("passed the gate; deploy comes in step 8b\n")

    @task
    def reject() -> None:
        reasons = json.loads((run_dir() / "gate.json").read_text())["reasons"]
        print(f"rejected, the champion stays: {'; '.join(reasons)}")

    data = build_data(drift_sensor())
    gate(score(data, finetune(data))) >> [deploy(), reject()]


drift_retrain()
