"""Quality gate: may a candidate adapter replace the champion? Compares their saved
answers to the same test questions, question by question (fair checker + calculator,
as registry.scores), and says pass or reject with the reasons."""
import math

from training.qlora_financial.registry import scores
from training.qlora_financial.sample_eval import rescore, with_calculator


def sign_test(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test on the questions the two models disagree on: how likely
    a split at least this lopsided is if both were equally good (each disagreement a coin flip)."""
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def decide(candidate_rows: list, champion_rows: list, alpha: float = 0.05, max_drop: float = 0.02) -> dict:
    """Pass only if the candidate gets more questions right that the champion misses than the
    reverse, that split isn't luck (sign test p < alpha), and no source drops by more than
    max_drop. Both must have answered the same questions."""
    graded = [{r["id"]: r["greedy_ok"] for r in rescore(with_calculator(rows), "fair")}
              for rows in (candidate_rows, champion_rows)]
    if graded[0].keys() != graded[1].keys():
        raise ValueError("candidate and champion answered different questions")
    cand, champ = graded
    wins, losses = sum(cand[i] and not champ[i] for i in cand), sum(champ[i] and not cand[i] for i in cand)
    p = sign_test(wins, losses)
    s_cand, s_champ = scores(candidate_rows), scores(champion_rows)
    reasons = []
    if wins <= losses:
        reasons.append(f"not more right answers than the champion ({wins} vs {losses})")
    elif p >= alpha:
        reasons.append(f"could be luck: {wins} vs {losses}, sign test p = {p:.2g} (needs < {alpha})")
    for src in s_champ:
        if src not in ("n", "ALL") and (drop := s_champ[src] - s_cand.get(src, 0.0)) > max_drop:
            reasons.append(f"{src} drops {drop * 100:.1f} points (limit {max_drop * 100:g})")
    return {"passed": not reasons, "reasons": reasons, "only_candidate": wins, "only_champion": losses,
            "p": p, "candidate": s_cand, "champion": s_champ}
