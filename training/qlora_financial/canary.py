"""Canary: every few minutes, ask the served model the same 50 test questions and publish its
accuracy, so a broken or swapped model shows up on the dashboard (speed metrics can't see it).
"Normal" is the champion's own saved score on the same 50 questions."""
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from training.qlora_financial.calculator import apply_calculator
from training.qlora_financial.registry import load_rows
from training.qlora_financial.sample_eval import chat, load_questions, match_variant


def pick(questions: list, n: int = 50, seed: int = 0) -> list:
    """The canary questions: a fixed random n (same seed, same questions every run)."""
    return random.Random(seed).sample(questions, n)


def grade(questions: list, answers: list) -> float:
    """Share answered right, as the registry scores: the calculator redoes the written steps,
    then the fair checker compares the final number with the gold answer."""
    right = [match_variant(apply_calculator(a), q["gold"], "fair") for q, a in zip(questions, answers, strict=True)]
    return sum(right) / len(right)


def run_once(url: str, model: str, questions: list, workers: int = 8, timeout: float = 60) -> dict:
    """Ask every canary question once (greedy, as the saved answers were made) and grade the
    answers. A question the server fails to answer counts as an error, not as wrong, so a
    server that's down shows up as errors (and a stale canary), not as a bad model. A round is
    scored only if every question was answered: "normal" is measured on all of them, and a score
    from the few that got through would be noise. `timeout` (seconds per question) keeps a hung
    server from stretching a round past the next one."""
    def ask(q):
        try:
            return chat(url, q["messages"], 1, 0.0, model=model, timeout=timeout)[0]
        except Exception:
            return None
    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, questions))
    errors = sum(a is None for a in answers)
    return {"accuracy": grade(questions, answers) if errors == 0 else None,
            "answered": len(questions) - errors, "errors": errors}


def expected_accuracy(saved_rows: list, questions: list) -> float:
    """"Normal": the champion's saved greedy answers to the same questions, graded the same way."""
    saved = {r["id"]: r["greedy"] for r in saved_rows}
    return grade(questions, [saved[q["id"]] for q in questions])


def make_metrics(registry, expected: float, now: float = None) -> dict:
    m = {"accuracy": Gauge("canary_accuracy", "Share of canary questions answered right (last round; NaN = not every question was answered, or none yet)",
                           registry=registry),
         "errors": Counter("canary_errors", "Canary questions the server failed to answer", registry=registry),
         "last_run": Gauge("canary_last_run_timestamp_seconds", "When a round last had every question answered (at launch: the start time)", registry=registry)}
    Gauge("canary_expected_accuracy", "The champion's saved score on the same questions", registry=registry).set(expected)
    m["accuracy"].set(float("nan"))  # not 0 before the first answers: 0 would look like a broken model
    m["last_run"].set(time.time() if now is None else now)  # stale wait starts at launch, not in 1970
    return m


def record(metrics: dict, result: dict, now: float = None) -> None:
    """A round without a score (some questions unanswered) clears accuracy (NaN) and leaves the timestamp alone, so the canary
    goes stale; an old low score kept up through an outage would page as if every round were bad."""
    metrics["errors"].inc(result["errors"])
    if result["accuracy"] is None:
        metrics["accuracy"].set(float("nan"))
    else:
        metrics["accuracy"].set(result["accuracy"])
        metrics["last_run"].set(time.time() if now is None else now)


def main():
    ap = argparse.ArgumentParser(description="Ask the served model the canary questions every few minutes; "
                                             "publish the accuracy on /metrics for Prometheus.")
    ap.add_argument("--url", default="http://localhost:8000", help="the OpenAI-style server (vLLM)")
    ap.add_argument("--model", default="v3", help="the served model name to ask")
    ap.add_argument("--data-dir", default="training/qlora_financial/data/v3", help="its test.jsonl")
    ap.add_argument("--expected-from", help="saved answers that define normal (default: the registry's champion)")
    ap.add_argument("--tracking-uri", default="sqlite:///training/qlora_financial/outputs/mlflow.db")
    ap.add_argument("--every", type=float, default=300, help="seconds between rounds")
    ap.add_argument("--port", type=int, default=9101, help="where /metrics is served")
    args = ap.parse_args()
    if args.expected_from is None:
        from training.qlora_financial.promote import champion_samples
        args.expected_from = champion_samples(args.tracking_uri)
    questions = pick(load_questions(args.data_dir))
    expected = expected_accuracy(load_rows(args.expected_from), questions)
    registry = CollectorRegistry()
    metrics = make_metrics(registry, expected)
    start_http_server(args.port, registry=registry)
    print(f"{len(questions)} canary questions, normal {expected:.2f} ({args.expected_from}); "
          f"/metrics on :{args.port}, a round every {args.every:g} s", flush=True)
    while True:
        started = time.monotonic()
        result = run_once(args.url, args.model, questions)
        record(metrics, result)
        acc = "none" if result["accuracy"] is None else f"{result['accuracy']:.2f}"
        print(f"{time.strftime('%H:%M:%S')} accuracy {acc} (answered {result['answered']}, errors {result['errors']})",
              flush=True)
        time.sleep(max(0.0, args.every - (time.monotonic() - started)))  # a round starts every --every s, however long it took


if __name__ == "__main__":
    main()
