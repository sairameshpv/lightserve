"""promote.py: the quality gate, on made-up and saved answers. Pure Python, no GPU."""
from pathlib import Path

import pytest

from training.qlora_financial.promote import champion_samples, decide, main, sign_test
from training.qlora_financial.registry import REPO, load_rows

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"  # gitignored: these tests skip where it's absent (CI)


def test_sign_test_by_hand():
    assert sign_test(5, 5) == 1.0  # an even split is no evidence either way
    assert sign_test(0, 5) == 2 / 2**5  # 5 of 5 one way: 1/32, doubled for two-sided
    assert sign_test(1, 9) == sign_test(9, 1) == 2 * 11 / 2**10  # C(10,0) + C(10,1) = 1 + 10 = 11
    assert sign_test(0, 0) == 1.0  # no disagreements, nothing to test
    assert f"{sign_test(151, 47):.0e}" == "6e-14"  # v3 vs v2 overall, the README's number


def answers(right_by_source: dict) -> list:
    """Rows of 100 questions per source, the first `right` of them answered correctly."""
    return [{"id": f"{src}{i}", "source": src, "gold": "1.0", "samples": [],
             "greedy": "Answer: 1.0" if i < right else "Answer: 2.0"}
            for src, right in right_by_source.items() for i in range(100)]


def test_one_source_dropping_3_points_rejects_despite_a_big_overall_win():
    d = decide(answers({"A": 100, "B": 97}), answers({"A": 50, "B": 100}))
    assert (d["only_candidate"], d["only_champion"]) == (50, 3) and d["p"] < 1e-10
    assert not d["passed"]
    assert d["reasons"] == ["B drops 3.0 points (limit 2)"]


def test_no_better_is_rejected():
    d = decide(answers({"A": 60}), answers({"A": 60}))
    assert not d["passed"] and d["reasons"] == ["not more right answers than the champion (0 vs 0)"]


@pytest.mark.skipif(not (OUTPUTS / "v3" / "samples.jsonl").exists(), reason="saved answers not here")
def test_real_v3_beats_v2_and_not_the_reverse():
    v2, v3 = (load_rows(OUTPUTS / v / "samples.jsonl") for v in ("v2", "v3"))
    d = decide(v3, v2)
    assert d["passed"] and (d["only_candidate"], d["only_champion"]) == (151, 47) and d["reasons"] == []
    back = decide(v2, v3)
    assert not back["passed"]
    assert "not more right answers than the champion (47 vs 151)" in back["reasons"]
    assert "ConvFinQA drops 17.3 points (limit 2)" in back["reasons"]  # 86.528 - 69.257


def test_command_error_exits_2_not_1(monkeypatch, tmp_path):
    """1 means "reject"; a crash (here a missing file) must not look like one."""
    monkeypatch.setattr("sys.argv", ["promote", "--candidate", str(tmp_path / "missing.jsonl"),
                                     "--champion", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2


def test_a_drop_of_exactly_2_points_is_allowed():
    """45 -> 44 right of 50 is exactly 2.0 points, at the limit, not over it (float error makes it 2.0000000000000018)."""
    b = lambda right: [r for r in answers({"B": right}) if int(r["id"][1:]) < 50]  # a 50-question source
    d = decide(answers({"A": 100}) + b(44), answers({"A": 50}) + b(45))
    assert d["passed"] and d["reasons"] == []


def test_champion_samples_resolves_a_repo_relative_path(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # a throwaway database, never the real one
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    client.create_registered_model("finance-qa")
    mv = client.create_model_version("finance-qa", source="x", tags={"samples": "training/v9/samples.jsonl"})
    client.set_registered_model_alias("finance-qa", "champion", mv.version)
    assert champion_samples(uri) == str(REPO / "training" / "v9" / "samples.jsonl")
