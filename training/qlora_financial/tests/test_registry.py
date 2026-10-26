"""registry.py: scores, lineage and registration on made-up data. Pure Python + MLflow, no GPU."""
import hashlib
import json

import pytest

from training.qlora_financial.registry import REPO, lineage, register, repo_path, scores


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
    assert v2.tags["git_commit"] == "bbb" and v2.source == str(adapter.resolve())
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 1
    run = client.get_run(v1.run_id)
    assert run.data.metrics["acc.FinQA"] == 0.5
    assert run.inputs.dataset_inputs[0].dataset.name == "train.jsonl"

    register("newer", git_commit="ccc", champion=True, **common)
    assert int(client.get_model_version_by_alias("finance-qa", "champion").version) == 3


def test_repo_path_is_relative_inside_the_repo(tmp_path):
    """Stored paths must work on any machine with the repo, not just this one."""
    assert repo_path(REPO / "training" / "qlora_financial" / "outputs" / "v3" / "samples.jsonl") == \
        "training/qlora_financial/outputs/v3/samples.jsonl"
    assert repo_path("training/qlora_financial/data/v3") == "training/qlora_financial/data/v3"  # from the repo root
    assert repo_path(tmp_path / "x.jsonl") == str(tmp_path / "x.jsonl")  # outside the repo: kept absolute
