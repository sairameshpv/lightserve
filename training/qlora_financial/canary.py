"""Canary: every few minutes, ask the served model the same 50 test questions and publish its
accuracy, so a broken or swapped model shows up on the dashboard (speed metrics can't see it).
"Normal" is the champion's own saved score on the same 50 questions."""
import random
from concurrent.futures import ThreadPoolExecutor

from training.qlora_financial.calculator import apply_calculator
from training.qlora_financial.sample_eval import chat, match_variant


def pick(questions: list, n: int = 50, seed: int = 0) -> list:
    """The canary questions: a fixed random n (same seed, same questions every run)."""
    return random.Random(seed).sample(questions, n)


def grade(questions: list, answers: list) -> float:
    """Share answered right, as the registry scores: the calculator redoes the written steps,
    then the fair checker compares the final number with the gold answer."""
    right = [match_variant(apply_calculator(a), q["gold"], "fair") for q, a in zip(questions, answers, strict=True)]
    return sum(right) / len(right)


def run_once(url: str, model: str, questions: list, workers: int = 8) -> dict:
    """Ask every canary question once (greedy, as the saved answers were made) and grade the
    answers. A question the server fails to answer counts as an error, not as wrong, so a
    server that's down shows up as errors (and a stale canary), not as a bad model."""
    def ask(q):
        try:
            return chat(url, q["messages"], 1, 0.0, model=model)[0]
        except Exception:
            return None
    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(ask, questions))
    done = [(q, a) for q, a in zip(questions, answers) if a is not None]
    return {"accuracy": grade(*zip(*done)) if done else None, "answered": len(done),
            "errors": len(questions) - len(done)}
