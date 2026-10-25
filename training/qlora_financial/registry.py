"""Model registry for the finance QA adapters: each adapter becomes a numbered
version of `finance-qa` in MLflow, with its scores and where it came from
(data hash, code commit, base model), and a `champion` alias on the one being
served.

Scores are the README's headline numbers: greedy accuracy under the fair
checker after the calculator, recomputed from the saved answers (samples.jsonl)."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from training.qlora_financial.sample_eval import rescore, with_calculator


def load_rows(samples_path) -> list:
    with open(samples_path) as f:
        return [json.loads(line) for line in f]


def scores(rows: list) -> dict:
    """{"n", "ALL", <source>: accuracy} for greedy answers, fair checker + calculator."""
    by_source = defaultdict(list)
    for r in rescore(with_calculator(rows), "fair"):
        by_source[r["source"]].append(r["greedy_ok"])
    every = [ok for oks in by_source.values() for ok in oks]
    return {"n": len(every), "ALL": sum(every) / len(every),
            **{src: sum(oks) / len(oks) for src, oks in sorted(by_source.items())}}


def lineage(adapter_dir, data_dir, git_commit: str) -> dict:
    """Where a version came from: base model and LoRA settings (its adapter_config.json),
    the code commit, and a content hash + row count per data split -- v2's and v3's
    train.jsonl have the same name and 34,141 rows each, so only the hash tells them apart."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    lin = {"base_model": cfg["base_model_name_or_path"], "lora_r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
           "git_commit": git_commit, "data_dir": str(data_dir)}
    for split in ("train", "val", "test"):
        raw = (Path(data_dir) / f"{split}.jsonl").read_bytes()
        lin[f"data.{split}.sha256"] = hashlib.sha256(raw).hexdigest()
        lin[f"data.{split}.rows"] = raw.count(b"\n")
    return lin


def register(version_name: str, adapter_dir, samples_path, data_dir, git_commit: str, artifact_root,
             champion: bool = False, model: str = "finance-qa") -> int:
    """Register one adapter as the next version of `model`: a small "registration" run holds
    the scores (metrics), lineage (params) and the training data (dataset input); the version
    points at the adapter folder and carries the same facts as tags. Returns the version number.
    Uses whatever MLflow tracking URI is set (the caller picks the database)."""
    import mlflow
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset
    from mlflow.exceptions import MlflowException

    s, lin = scores(load_rows(samples_path)), lineage(adapter_dir, data_dir, git_commit)
    facts = {"version_name": version_name, "samples": str(Path(samples_path).resolve()), **lin,
             **{f"acc.{k}": v for k, v in s.items() if k != "n"}, "n_scored": s["n"]}
    if mlflow.get_experiment_by_name("finance-qa-registry") is None:
        mlflow.create_experiment("finance-qa-registry", artifact_location=Path(artifact_root).resolve().as_uri())
    mlflow.set_experiment("finance-qa-registry")
    with mlflow.start_run(run_name=f"register-{version_name}") as run:
        mlflow.log_metrics({k: v for k, v in facts.items() if k.startswith("acc.")})
        mlflow.log_params({k: v for k, v in facts.items() if not k.startswith("acc.")})
        train = Path(data_dir) / "train.jsonl"  # the dataset field holds 36 chars, so the hash is cut to 32
        mlflow.log_input(MetaDataset(resolve_dataset_source(str(train.resolve())), name=train.name,
                                     digest=lin["data.train.sha256"][:32]), context="training")
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(model)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    mv = client.create_model_version(model, source=str(Path(adapter_dir).resolve()), run_id=run.info.run_id,
                                     tags={k: str(v) for k, v in facts.items()})
    if champion:
        client.set_registered_model_alias(model, "champion", mv.version)
    return int(mv.version)


def main():
    ap = argparse.ArgumentParser(description="Register a finance-qa adapter version in MLflow.")
    ap.add_argument("--version-name", required=True, help="e.g. v3")
    ap.add_argument("--adapter", required=True, help="adapter folder (with adapter_config.json)")
    ap.add_argument("--samples", required=True, help="its saved test answers (sample_eval's samples.jsonl)")
    ap.add_argument("--data-dir", required=True, help="the train/val/test.jsonl it was trained on")
    ap.add_argument("--git-commit", required=True, help="the code commit that trained it")
    ap.add_argument("--champion", action="store_true", help="move the champion alias to this version")
    ap.add_argument("--tracking-uri", default="sqlite:///training/qlora_financial/outputs/mlflow.db")
    ap.add_argument("--artifact-root", default="training/qlora_financial/outputs/mlruns")
    args = ap.parse_args()
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    n = register(args.version_name, args.adapter, args.samples, args.data_dir, args.git_commit,
                 args.artifact_root, champion=args.champion)
    print(f"finance-qa version {n} = {args.version_name}" + (" (champion)" if args.champion else ""))


if __name__ == "__main__":
    main()
