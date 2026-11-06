"""registry.py: scores, lineage and registration on made-up data. Pure Python + MLflow, no GPU."""
import hashlib
import json

import pytest

from training.qlora_financial.registry import REPO, finish_champion, lineage, register, repo_path, scores


def row(id, source, greedy, gold):
    return {"id": id, "source": source, "greedy": greedy, "gold": gold, "samples": []}


def test_scores_fair_checker_with_calculator():
    rows = [
        row(1, "FinQA", "Step 1: divide 151 and 211.\nAnswer: 0.7168", "0.71564"),  # wrong as written, right once recalculated
        row(2, "FinQA", "Answer: 12.0", "10.0"),  # plain wrong
        row(3, "TAT-QA", "Answer: 39.34", "39.3"),  # fair: a more precise answer that rounds to the gold
        row(4, "TAT-QA", "Answer: 2019", "2018"),  # years must match exactly
    ]
    s = scores(rows)
    assert s == {"n": 4, "ALL": 0.5, "FinQA": 0.5, "TAT-QA": 0.5}


def test_lineage(tmp_path):
    adapter, data = tmp_path / "adapter", tmp_path / "data"
    adapter.mkdir(), data.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct", "r": 16, "lora_alpha": 32}))
    for split, n in (("train", 3), ("val", 1), ("test", 2)):
        (data / f"{split}.jsonl").write_text("".join(f'{{"i": {i}}}\n' for i in range(n)))
    lin = lineage(adapter, data, git_commit="abc1234")
    assert lin["base_model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert (lin["lora_r"], lin["lora_alpha"], lin["git_commit"]) == (16, 32, "abc1234")
    assert (lin["data.train.rows"], lin["data.val.rows"], lin["data.test.rows"]) == (3, 1, 2)
    assert lin["data.train.sha256"] == hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()
    assert lin["data_dir"] == str(data)
    (data / "val.jsonl").write_text('{"i": 0}\n{"i": 1}')  # no line break after the last row
    assert lineage(adapter, data, git_commit="abc1234")["data.val.rows"] == 2


def test_register_two_versions_and_move_champion(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")  # a throwaway database, never the real one
    adapter, data = tmp_path / "adapter", tmp_path / "data"
    adapter.mkdir(), data.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base", "r": 8, "lora_alpha": 16}))
    for split in ("train", "val", "test"):
        (data / f"{split}.jsonl").write_text('{"i": 0}\n')
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps(r) + "\n" for r in [row(1, "FinQA", "Answer: 10.0", "10.0"),
                                                              row(2, "FinQA", "Answer: 12.0", "10.0")]))
    common = dict(adapter_dir=adapter, samples_path=samples, data_dir=data, artifact_root=tmp_path / "artifacts")
    assert register("old", git_commit="aaa", champion=True, **common) == 1
    assert register("new", git_commit="bbb", **common) == 2

    client = mlflow.MlflowClient()
    v1, v2 = (client.get_model_version("finance-qa", str(n)) for n in (1, 2))
    assert (v1.tags["version_name"], v1.tags["acc.ALL"], v1.tags["git_commit"]) == ("old", "0.5", "aaa")
    assert v2.tags["git_commit"] == "bbb" and v2.tags["adapter_path"] == str(adapter.resolve())
    # the adapter itself is stored with the registration run, and the version points there: an
    # MLflow server refuses a plain folder as a version's source (found by the 8j kind rehearsal)
    assert v2.source.endswith("/adapter") and v2.source.startswith(client.get_run(v2.run_id).info.artifact_uri)
    assert [a.path for a in client.list_artifacts(v2.run_id, "adapter")] == ["adapter/adapter_config.json"]
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 1
    run = client.get_run(v1.run_id)
    assert run.data.metrics["acc.FinQA"] == 0.5
    assert run.inputs.dataset_inputs[0].dataset.name == "train.jsonl"

    register("newer", git_commit="ccc", champion=True, **common)
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 3

    runs_before = len(mlflow.search_runs(search_all_experiments=True))
    with pytest.raises(ValueError, match="'old' is already finance-qa version 1"):
        register("old", git_commit="aaa", **common)  # run twice by mistake
    assert len(mlflow.search_runs(search_all_experiments=True)) == runs_before  # refused before writing anything
    assert len(client.search_model_versions("name='finance-qa'")) == 3

    register("rolled-back", git_commit="ddd", tags={"status": "rejected-after-deploy"}, **common)
    assert client.get_model_version("finance-qa", "4").tags["status"] == "rejected-after-deploy"
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 3  # not champion


def test_a_registration_cut_off_before_the_champion_label_can_be_finished(tmp_path, monkeypatch):
    """MLflow dies after the version exists but before the label moves (8j): the retry is refused
    as "already registered", so it must still finish the label, or the model is half-registered."""
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    adapter, data = tmp_path / "adapter", tmp_path / "data"
    adapter.mkdir(), data.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base", "r": 8, "lora_alpha": 16}))
    for split in ("train", "val", "test"):
        (data / f"{split}.jsonl").write_text('{"i": 0}\n')
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(row(1, "FinQA", "Answer: 10.0", "10.0")) + "\n")
    common = dict(adapter_dir=adapter, samples_path=samples, data_dir=data, artifact_root=tmp_path / "artifacts")
    register("old", git_commit="aaa", champion=True, **common)

    real = mlflow.MlflowClient.set_registered_model_alias
    def mlflow_dies(*a, **k):
        raise ConnectionError("MLflow went away")
    monkeypatch.setattr(mlflow.MlflowClient, "set_registered_model_alias", mlflow_dies)
    with pytest.raises(ConnectionError):
        register("new", git_commit="bbb", champion=True, **common)
    monkeypatch.setattr(mlflow.MlflowClient, "set_registered_model_alias", real)   # MLflow is back

    client = mlflow.MlflowClient()
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 1  # half-registered
    with pytest.raises(ValueError, match="already"):
        register("new", git_commit="bbb", champion=True, **common)  # the retry is refused...
    assert finish_champion("new") == 2                              # ...so it finishes the label
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 2
    assert finish_champion("new") == 2                              # a second retry: nothing more to do
    assert len(client.search_model_versions("name='finance-qa'")) == 2


@pytest.fixture
def mlflow_server(tmp_path):
    """A real MLflow server (like the cluster's): it, unlike a local database file, checks a
    version's source. Throwaway database and artifact folder."""
    pytest.importorskip("mlflow")
    import os, socket, subprocess, sys, time, urllib.request
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    env = {**os.environ, "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false"}
    proc = subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(port),
                             "--backend-store-uri", f"sqlite:///{tmp_path / 'server.db'}",
                             "--artifacts-destination", str(tmp_path / "server-artifacts"), "--workers", "1"],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("MLflow server didn't start")
    yield url, tmp_path / "server-artifacts"
    proc.kill()
    proc.wait()


def test_register_against_an_mlflow_server_like_the_clusters(tmp_path, mlflow_server):
    """Kind: "INVALID_PARAMETER_VALUE: Invalid model version source" on every registration. And the
    adapter must land in the SERVER's storage (the cluster's disk), not the client pod's /tmp."""
    import mlflow
    url, server_store = mlflow_server
    mlflow.set_tracking_uri(url)
    adapter, data = tmp_path / "adapter", tmp_path / "data"
    adapter.mkdir(), data.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base", "r": 8, "lora_alpha": 16}))
    for split in ("train", "val", "test"):
        (data / f"{split}.jsonl").write_text('{"i": 0}\n')
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(row(1, "FinQA", "Answer: 10.0", "10.0")) + "\n")
    client_side = tmp_path / "client-artifacts"   # what a client passes; a server must not use it
    v = register("srv", adapter_dir=adapter, samples_path=samples, data_dir=data, git_commit="a",
                 artifact_root=client_side, champion=True)
    client = mlflow.MlflowClient()
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == v == 1
    assert not client_side.exists()
    assert list(server_store.rglob("adapter_config.json"))     # stored by the server


def test_champion_info_reads_versions_registered_before_adapter_path_existed(tmp_path):
    """The real registry's v1/v2 were registered with source = the repo folder and no adapter_path."""
    mlflow = pytest.importorskip("mlflow")
    from mlops.retrain import champion_info
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    client.create_registered_model("finance-qa")
    mv = client.create_model_version("finance-qa", source="training/qlora_financial/outputs/v3",
                                     tags={"version_name": "v3", "data_dir": "data/v3", "samples": "s.jsonl"})
    client.set_registered_model_alias("finance-qa", "champion", mv.version)
    assert champion_info(uri)["adapter"] == str(REPO / "training/qlora_financial/outputs/v3")


def test_repo_path_is_relative_inside_the_repo(tmp_path):
    """Stored paths must work on any machine with the repo, not just this one."""
    assert repo_path(REPO / "training" / "qlora_financial" / "outputs" / "v3" / "samples.jsonl") == \
        "training/qlora_financial/outputs/v3/samples.jsonl"
    assert repo_path("training/qlora_financial/data/v3") == "training/qlora_financial/data/v3"  # from the repo root
    assert repo_path(tmp_path / "x.jsonl") == str(tmp_path / "x.jsonl")  # outside the repo: kept absolute

