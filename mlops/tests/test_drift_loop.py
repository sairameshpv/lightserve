"""The whole retraining loop, end to end: the Airflow DAG run by dag.test() in dry-run mode on
made-up data, against one fake server playing vLLM (adapters), Prometheus (the canary) and
Alertmanager. Four endings: no drift; a good candidate (new champion); a bad one (gate
rejects); a secretly bad one (passes the gate, answers wrong live: rolled back).
Each scenario: a throwaway MLflow registry; Airflow in its own process (its settings are read
at import)."""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE = "\n".join("Revenue\t100\t200" for _ in range(4))
TEXT = " ".join(["the company reported results for the year"] * 40)


def chat_row(i, doc, answer):
    return {"source": "FinQA" if i % 2 else "TAT-QA",
            "messages": [{"role": "system", "content": ""},
                         {"role": "user", "content": f"{doc}\nQuestion: what was it in {2000 + i}?"},
                         {"role": "assistant", "content": answer}]}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def answers(ids, wrong=()):
    """Saved greedy answers to test question i (right answer "Answer: i")."""
    return [{"id": i, "source": "FinQA" if i % 2 else "TAT-QA", "gold": f"Answer: {i}",
             "greedy": "Answer: -1" if i in wrong else f"Answer: {i}", "samples": []} for i in ids]


class Fakes(ThreadingHTTPServer):
    """vLLM: /v1/models, load/unload adapters. Prometheus: the canary asks whatever the pointer
    names, a new round on every look; a model in `wrong_live` scores 0, others 1.
    Alertmanager: records alerts."""
    request_queue_size = 64

    def __init__(self, pointer):
        self.pointer, self.adapters, self.wrong_live, self.alerts = pointer, {"v3"}, set(), []

        class Handler(BaseHTTPRequestHandler):
            def send(self, code, body):
                self.send_response(code)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/v1/models":
                    return self.send(200, {"data": [{"id": m} for m in ["base", *fakes.adapters]]})
                query = parse_qs(url.query)["query"][0]
                asked, since = fakes.canary_model()
                value = {"canary_accuracy": 0.0 if asked in fakes.wrong_live else 1.0,
                         "canary_expected_accuracy": 1.0,
                         "canary_last_run_timestamp_seconds": time.time()}.get(query)
                if query == f'canary_model_info{{model="{asked}"}}':
                    value = since
                result = [] if value is None else [{"metric": {}, "value": [time.time(), str(value)]}]
                self.send(200, {"status": "success", "data": {"resultType": "vector", "result": result}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/load_lora_adapter":
                    fakes.adapters.add(body["lora_name"])
                elif self.path == "/v1/unload_lora_adapter":
                    fakes.adapters.discard(body["lora_name"])
                elif self.path == "/api/v2/alerts":
                    fakes.alerts += body
                self.send(200, {})

            def log_message(self, *args):
                pass

        fakes = self
        super().__init__(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server_address[1]}"

    def canary_model(self):
        """The model the canary asks (the pointer's, else the champion) and since when."""
        if not self.pointer.exists():
            return "v3", 0.0
        return json.loads(self.pointer.read_text())["model"], self.pointer.stat().st_mtime


@pytest.fixture(scope="module")
def airflow_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("airflow")
    env = {**os.environ, "AIRFLOW_HOME": str(home), "AIRFLOW__CORE__LOAD_EXAMPLES": "False"}
    subprocess.run([sys.executable, "-m", "airflow", "db", "migrate"], env=env, check=True, capture_output=True)
    return home


@pytest.fixture
def world(tmp_path):
    """Champion "v3" in a throwaway registry: trained on 100 rows (35% tables), 40 test questions,
    right on all but 10 of them."""
    import mlflow
    from training.qlora_financial.registry import register
    data, adapter = tmp_path / "data", tmp_path / "adapter"
    data.mkdir(), adapter.mkdir()
    train = [chat_row(i, TABLE if i < 35 else TEXT, "Answer: 1") for i in range(100)]
    jsonl(data / "train.jsonl", train)
    jsonl(data / "val.jsonl", train[:5])
    jsonl(data / "test.jsonl", [chat_row(i, TEXT, f"Answer: {i}") for i in range(40)])
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "b", "r": 16, "lora_alpha": 32}))
    jsonl(tmp_path / "v3_answers.jsonl", answers(range(40), wrong=range(0, 40, 4)))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    register("v3", adapter, tmp_path / "v3_answers.jsonl", data, "test", tmp_path / "artifacts", champion=True)
    fakes = Fakes(tmp_path / "runs" / "current.json")
    threading.Thread(target=fakes.serve_forever, daemon=True).start()
    yield {"tmp": tmp_path, "uri": uri, "fakes": fakes, "train": train}
    fakes.shutdown()


def run_dag(world, airflow_home, window_prompts, dry_answers):
    tmp, fakes = world["tmp"], world["fakes"]
    jsonl(tmp / "window.jsonl", [{"prompt": p} for p in window_prompts])
    jsonl(tmp / "dry_answers.jsonl", dry_answers)
    env = {**os.environ, "AIRFLOW_HOME": str(airflow_home), "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
           "AIRFLOW__CORE__DAGS_FOLDER": str(REPO / "mlops/dags"), "PYTHONPATH": str(REPO),
           "MLFLOW_DISABLE_AGENT_HINT": "1", "MLOPS_DRY_RUN": "1", "MLOPS_RUNS": str(tmp / "runs"),
           "MLOPS_WINDOW": str(tmp / "window.jsonl"), "MLOPS_DRY_ANSWERS": str(tmp / "dry_answers.jsonl"),
           "MLOPS_TRACKING_URI": world["uri"], "MLOPS_ARTIFACT_ROOT": str(tmp / "artifacts"),
           "MLOPS_N_LIKE": "20", "MLOPS_SENSOR_TIMEOUT": "0", "MLOPS_VLLM_URL": fakes.url,
           "MLOPS_PROMETHEUS_URL": fakes.url, "MLOPS_ALERTMANAGER_URL": fakes.url,
           "MLOPS_WATCH_MINUTES": "0.05", "MLOPS_WATCH_POLL": "0.5", "MLOPS_WATCH_STALE": "60"}
    code = ("import json, sys; sys.path.insert(0, 'mlops/dags'); from drift_retrain import drift_retrain; "
            "r = drift_retrain().test(); "
            "print('STATES', json.dumps({ti.task_id: str(ti.state) for ti in r.get_task_instances()}))")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO, capture_output=True, text=True, timeout=300)
    lines = [line for line in out.stdout.splitlines() if line.startswith("STATES ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    return json.loads(lines[-1][len("STATES "):])


def registry(world):
    import mlflow
    client = mlflow.MlflowClient(tracking_uri=world["uri"])
    versions = {v.tags["version_name"]: v.tags for v in client.search_model_versions("name='finance-qa'")}
    return versions, client.get_model_version_by_alias("finance-qa", "champion").tags["version_name"]


def drifted():
    """Traffic that is 80% tables (training: 35%)."""
    return [chat_row(i, TABLE if i < 80 else TEXT, "")["messages"][1]["content"] for i in range(100, 200)]


def test_no_drift_nothing_retrains(world, airflow_home):
    states = run_dag(world, airflow_home, [r["messages"][1]["content"] for r in world["train"]], answers(range(40)))
    assert set(states.values()) == {"skipped"}
    assert not (world["tmp"] / "runs").exists() and registry(world)[1] == "v3"
    assert world["fakes"].adapters == {"v3"}


def test_good_candidate_becomes_champion(world, airflow_home):
    states = run_dag(world, airflow_home, drifted(), answers(range(40)))  # the 10 misses fixed
    assert states["deploy"] == "success" and states["register_champion"] == "success"
    assert states["reject"] == "skipped" and states["rollback"] == "skipped"
    versions, champion = registry(world)
    assert champion != "v3" and versions[champion]["pool_data_dir"] == str(world["tmp"] / "data")
    assert world["fakes"].adapters == {champion}  # v3 unloaded
    assert json.loads((world["tmp"] / "runs/current.json").read_text())["model"] == champion


def test_bad_candidate_is_rejected_by_the_gate(world, airflow_home):
    states = run_dag(world, airflow_home, drifted(), answers(range(40), wrong=range(1, 40, 4)))
    assert states["reject"] == "success" and states["deploy"] == "skipped"
    assert registry(world)[1] == "v3" and len(registry(world)[0]) == 1  # nothing registered
    assert world["fakes"].adapters == {"v3"}  # candidate unloaded
    assert not (world["tmp"] / "runs/current.json").exists()  # never switched


def test_secretly_bad_candidate_is_rolled_back(world, airflow_home):
    """Good saved answers (the gate passes), wrong answers live (the canary sees it)."""
    fakes = world["fakes"]

    class AllNew(set):  # any model but the champion answers wrong live
        def __contains__(self, model):
            return model != "v3"
    fakes.wrong_live = AllNew()
    states = run_dag(world, airflow_home, drifted(), answers(range(40)))
    assert states["deploy"] == "success" and states["rollback"] == "success"
    assert states["register_champion"] == "skipped"
    versions, champion = registry(world)
    assert champion == "v3"
    rolled = [name for name, tags in versions.items() if tags.get("status") == "rejected-after-deploy"]
    assert len(rolled) == 1 and "2 rounds in a row" in versions[rolled[0]]["rollback_reason"]
    assert json.loads((world["tmp"] / "runs/current.json").read_text())["model"] == "v3"  # pointer back
    assert fakes.adapters == {"v3"}  # the bad one unloaded
    assert [a["labels"]["alertname"] for a in fakes.alerts] == ["ModelRolledBack"]
