"""Best-of-N: the same question answered N times with different random choices, then a vote.

All N agent runs are sent at once, so vLLM serves them as one batch. Each run uses its own seed
(0..N-1); both models get the same seeds, settings and passages, so only the model differs.
The vote is on each run's final number, read the way the finance model's checker reads it.
"""
import math
import re
from concurrent.futures import ThreadPoolExecutor

from rag import agent
from training.qlora_financial.sample_eval import _parts


# "in fiscal 2019", "for fiscal year 2018", "during 2019", "FY2019": a year said after the answer.
_YEAR = re.compile(r"\b(?:(?:in|for|during)\s+(?:fiscal\s+(?:year\s+)?)?|fiscal\s+(?:year\s+)?|FY\s*)(?:19|20)\d\d\b", re.I)


def _read(text):
    """(number, scale, decimals) of an answer, or None. The finance checker reads the LAST number,
    so a year said after the answer is dropped first ("$641 million in fiscal 2019" is $641
    million), unless nothing else is left ("2019" stays 2019). A percent is read as its fraction."""
    if not text:
        return None
    stripped = _YEAR.sub("", text)
    parts = _parts(stripped) or _parts(text)
    if parts and re.search(r"\d\s*(%|percent)", stripped, re.I):   # 2% is 0.02, so "-0.02" and "-2%" agree
        parts = (parts[0] / 100, parts[1], parts[2] + 2)
    return parts


def _same(a: tuple, b: tuple) -> bool:
    """Two (number, scale, decimals): within 1%, or equal once the more precise is rounded to the
    coarser one's decimals (-2.04% and -2%: the report rounds, the calculator doesn't). Years (whole,
    1900-2100, no scale word) must be equal: 2018 is within 1% of 2019 (the finance checker's rule)."""
    if any(x[1] == 1 and x[0].is_integer() and 1900 <= x[0] <= 2100 for x in (a, b)):
        return a[0] * a[1] == b[0] * b[1]
    if math.isclose(a[0] * a[1], b[0] * b[1], rel_tol=0.01, abs_tol=1e-6):
        return True
    d = min(a[2], b[2])                     # rounding to 0 says nothing: "0" is not a rounded 28%
    return a[1] == b[1] and round(a[0], d) == round(b[0], d) != 0


def same_answer(a, b) -> bool:
    """Do two answers give the same number, by the vote's rule? No number on either side: no."""
    pa, pb = _read(a), _read(b)
    return pa is not None and pb is not None and _same(pa, pb)


def vote(answers: list) -> dict:
    """Group the answers' final numbers; the biggest group wins, ties go to the first seen; runs
    with no number never win. -> {"answer" (the winning group's first text), "value", "votes",
    "n", "groups": [{"answer", "value", "votes", "runs"}] biggest first}."""
    groups = []
    for i, text in enumerate(answers):
        parts = _read(text)
        if parts is None:
            continue
        for g in groups:
            if _same(parts, g["parts"]):
                g["votes"] += 1
                g["runs"].append(i)
                break
        else:
            groups.append({"answer": text, "value": parts[0] * parts[1], "votes": 1, "runs": [i], "parts": parts})
    groups.sort(key=lambda g: -g["votes"])                 # stable: ties keep first-seen order
    for g in groups:
        del g["parts"]
    top = groups[0] if groups else {"answer": None, "value": None, "votes": 0}
    return {"answer": top["answer"], "value": top["value"], "votes": top["votes"], "n": len(answers), "groups": groups}


def best_of_n(question: str, chat, search_fn, n: int = 5, temperature: float = 0.7, top_p: float = 0.95,
              **run_kwargs) -> dict:
    """`chat(messages, temperature, top_p, seed) -> reply text` (one model). N agent runs at
    once, seeds 0..N-1. A run that fails (e.g. a server error) counts as no answer.
    -> {"vote": vote(...), "runs": [one per seed: agent.run's result + "seed" (+ "error")]}."""
    def one(seed):
        try:
            r = agent.run(question, lambda m: chat(m, temperature=temperature, top_p=top_p, seed=seed),
                          search_fn, **run_kwargs)
        except Exception as e:                               # one bad run must not sink the other N-1
            r = {"answer": None, "explanation": None, "stopped": "error", "steps": [], "passages": [], "error": str(e)}
        return {"seed": seed, **r}

    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(pool.map(one, range(n)))
    return {"vote": vote([r["answer"] for r in runs]), "runs": runs}
