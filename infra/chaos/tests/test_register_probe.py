"""Tests for infra/chaos/register_probe.py (experiment 6, MLflow down): it registers a test model
the pipeline's way (one retry after a wait; a retry finishes the champion label) and reports
what the registry ended up with. A throwaway MLflow database; needs mlflow (mlops venv)."""
import pytest

mlflow = pytest.importorskip("mlflow")
from infra.chaos import register_probe  # noqa: E402


def test_mlflow_up_registers_once_with_the_champion_and_reports_no_wait(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    r = register_probe.probe(uri, "chaos-1", retry_delay_s=0, work=tmp_path)
    assert r == {"first_try_failed_or_waited": False, "first_try": "ok", "versions": 1,
                 "champion_is_new": True, "facts_complete": True, "first_try_s": r["first_try_s"]}


def test_a_failed_first_try_is_retried_and_ends_complete(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    real, calls = register_probe.register, []
    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("MLflow is down")
        return real(*a, **k)
    monkeypatch.setattr(register_probe, "register", flaky)
    r = register_probe.probe(uri, "chaos-2", retry_delay_s=0, work=tmp_path)
    assert r["first_try"] == "failed: MLflow is down" and r["first_try_failed_or_waited"]
    assert (r["versions"], r["champion_is_new"], r["facts_complete"]) == (1, True, True)


def test_a_first_try_cut_off_before_the_label_is_finished_by_the_retry(tmp_path, monkeypatch):
    """The narrow case: the version exists, the label didn't move; the retry must finish it."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    real_alias = mlflow.MlflowClient.set_registered_model_alias
    calls = []
    def dies_once(self, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("MLflow went away")
        return real_alias(self, *a, **k)
    monkeypatch.setattr(mlflow.MlflowClient, "set_registered_model_alias", dies_once)
    r = register_probe.probe(uri, "chaos-3", retry_delay_s=0, work=tmp_path)
    assert (r["versions"], r["champion_is_new"], r["facts_complete"]) == (1, True, True)
