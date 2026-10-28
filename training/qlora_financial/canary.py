"""Canary: every few minutes, ask the served model the same 50 test questions and publish its
accuracy, so a broken or swapped model shows up on the dashboard (speed metrics can't see it).
"Normal" is the champion's own saved score on the same 50 questions."""
import random

from training.qlora_financial.calculator import apply_calculator
from training.qlora_financial.sample_eval import match_variant


def pick(questions: list, n: int = 50, seed: int = 0) -> list:
    """The canary questions: a fixed random n (same seed, same questions every run)."""
    return random.Random(seed).sample(questions, n)


def grade(questions: list, answers: list) -> float:
    """Share answered right, as the registry scores: the calculator redoes the written steps,
    then the fair checker compares the final number with the gold answer."""
    right = [match_variant(apply_calculator(a), q["gold"], "fair") for q, a in zip(questions, answers, strict=True)]
    return sum(right) / len(right)
