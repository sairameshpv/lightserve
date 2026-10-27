"""Query drift: do the questions being asked still look like the ones the model was
trained on? Each question becomes a few measurements taken from its text alone (real
users don't say which dataset their question resembles), and Evidently compares this
window's measurements with the training questions', column by column."""
import argparse
import json
import re
from pathlib import Path

RULER = re.compile(r"^[*=-]{10,}$")  # FinQA puts some tables between two of these
ROW_MARK = re.compile(r"^(• |Row \d+: )")  # FinQA's other row styles
# A measurement counts as changed above this normed Wasserstein distance. Evidently's default
# 0.1 alarmed on 4 of 5 unchanged windows of 500 test questions (their documents are ~13%
# longer than training's), so it was calibrated: the largest distance over 20 unchanged
# windows (seeds 10-29) was 0.209, rounded up to the next 0.05.
LIMIT = 0.25


def features(prompt: str) -> dict:
    """The measurements of one question. Prompts are a document (text and/or a table),
    then `Question: ...`; ConvFinQA puts earlier turns in between, so the document ends
    at the first `Question:` and the question starts after the last one."""
    document = prompt.split("Question:")[0]
    question = prompt.rsplit("Question:", 1)[1] if "Question:" in prompt else ""
    return {"question_words": len(question.split()), "document_words": len(document.split()),
            "table_rows": table_rows(prompt),
            "digit_share": sum(c.isdigit() for c in prompt) / max(len(prompt), 1)}


def table_rows(prompt: str) -> int:
    """Table rows in any of the sources' formats: ConvFinQA's web-table code (<tr>), TAT-QA's
    tabs, FinQA's ` | ` bars, `• `/`Row N: ` lines, or lines between two ruler lines. FinQA
    rows written as plain comma lines with no marker (about 4% of FinQA) aren't counted."""
    n, block = prompt.count("<tr>"), None
    for line in prompt.splitlines():
        if RULER.match(line.strip()):
            n, block = (n + block, None) if block is not None else (n, 0)  # closing ruler: count the block
        elif block is not None:
            block += 1
        elif "\t" in line or " | " in line or ROW_MARK.match(line):
            n += 1
    return n


def drift(reference_prompts: list, current_prompts: list, limit: float = LIMIT, alarm_share: float = 0.5) -> tuple:
    """Compare a window of questions with the reference (training) questions. Evidently tests each
    measurement (normed Wasserstein distance > `limit` = changed); the alarm is
    on when at least `alarm_share` of them changed. Returns (summary, Evidently snapshot for HTML)."""
    import pandas as pd
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    ref, cur = (pd.DataFrame([features(p) for p in prompts]) for prompts in (reference_prompts, current_prompts))
    dd = DataDefinition(numerical_columns=list(ref.columns))
    snap = Report([DataDriftPreset(drift_share=alarm_share, num_method="wasserstein", num_threshold=limit)]).run(
        Dataset.from_pandas(cur, data_definition=dd), Dataset.from_pandas(ref, data_definition=dd))
    metrics = snap.dict()["metrics"]
    drifted = [m["config"]["column"] for m in metrics[1:] if m["value"] > m["config"]["threshold"]]
    share = metrics[0]["value"]["share"]
    return {"drift_share": share, "drifted_columns": drifted, "dataset_drift": share >= alarm_share}, snap


def main():
    ap = argparse.ArgumentParser(description="Have the questions changed since training? Writes drift.json + report.html.")
    ap.add_argument("--current", required=True, help="a query window, one {\"prompt\": ...} per line (make_traffic.py)")
    ap.add_argument("--reference", default="training/qlora_financial/data/v3/train.jsonl", help="the champion's training data")
    ap.add_argument("--out", required=True, help="folder for drift.json and report.html")
    args = ap.parse_args()
    with open(args.reference) as f:
        reference = [json.loads(line)["messages"][1]["content"] for line in f]
    with open(args.current) as f:
        current = [json.loads(line)["prompt"] for line in f]
    summary, snap = drift(reference, current)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift.json").write_text(json.dumps(summary, indent=2) + "\n")
    snap.save_html(str(out / "report.html"))
    print(f"{summary['drift_share']:.0%} of measurements changed {summary['drifted_columns']} -> "
          f"{'ALARM' if summary['dataset_drift'] else 'no alarm'} ({out})")


if __name__ == "__main__":
    main()
