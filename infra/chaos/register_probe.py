"""Experiment 6 (MLflow down): register a TEST model the way the pipeline's register step does
(mlops/dags/drift_retrain.py register_once: one retry after a wait, like the DAG's default_args;
a retry that finds it "already registered" only finishes the champion label), then report what
the registry ended up with, as one JSON line for run_chaos.py's judge. Never touches finance-qa.

  python infra/chaos/register_probe.py --tracking-uri http://mlflow.mlops:80 --model chaos-123
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the unpacked code: training/...
from training.qlora_financial.registry import finish_champion, register  # noqa: E402

WAITED_S = 10      # a first try this slow means MLflow's client sat out (part of) the outage
NEEDED = ("version_name", "git_commit", "acc.ALL", "n_scored", "chaos")   # facts the version must carry


def _files(work):
    """A tiny adapter, data and saved answers: enough for register() to compute its facts."""
    adapter, data = work / "adapter", work / "data"
    adapter.mkdir(exist_ok=True), data.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base", "r": 8, "lora_alpha": 16}))
    for split in ("train", "val", "test"):
        (data / f"{split}.jsonl").write_text('{"i": 0}\n')
    samples = work / "samples.jsonl"
    samples.write_text(json.dumps({"id": 1, "source": "FinQA", "greedy": "Answer: 1", "gold": "1", "samples": []}) + "\n")
    return dict(adapter_dir=adapter, samples_path=samples, data_dir=data, artifact_root=work / "artifacts")


def probe(tracking_uri, model, retry_delay_s, work):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    files, name = _files(Path(work)), f"{model}-candidate"
    kw = dict(git_commit="chaos", champion=True, model=model, tags={"chaos": "mlflow-down"}, **files)
    t = time.time()
    try:
        register(name, **kw)
        first = "ok"
    except Exception as e:      # MLflow down: the first try fails (the DAG task would fail too)
        first = f"failed: {e}"
    first_s = round(time.time() - t, 1)
    if first != "ok":
        time.sleep(retry_delay_s)   # the DAG's retry_delay
        try:
            register(name, **kw)
        except ValueError as e:      # the first try got as far as creating the version
            if "is already" not in str(e):
                raise
            finish_champion(name, model=model)
    client = mlflow.MlflowClient()
    mine = [v for v in client.search_model_versions(f"name='{model}'") if v.tags.get("version_name") == name]
    try:
        champion = client.get_model_version_by_alias(model, "champion").version
    except mlflow.exceptions.MlflowException:
        champion = None
    return {"first_try_failed_or_waited": first != "ok" or first_s > WAITED_S, "first_try": first,
            "first_try_s": first_s, "versions": len(mine),
            "champion_is_new": bool(mine) and champion == mine[0].version,
            "facts_complete": bool(mine) and all(k in mine[0].tags for k in NEEDED)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-uri", required=True)
    ap.add_argument("--model", required=True, help="a test model name, e.g. chaos-<time>")
    ap.add_argument("--retry-delay-s", type=float, default=60, help="the DAG's retry_delay")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as work:
        print(json.dumps(probe(a.tracking_uri, a.model, a.retry_delay_s, work)), flush=True)


if __name__ == "__main__":
    main()
