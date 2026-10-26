"""Quality gate: may a candidate adapter replace the champion? Compares their saved
answers to the same test questions, question by question (fair checker + calculator,
as registry.scores), and says pass or reject with the reasons."""
import argparse
import math
import sys

from training.qlora_financial.registry import load_rows, scores
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
        # rounded: float error turns an exact 2-point drop into 2.0000000000000018, over the limit
        if src not in ("n", "ALL") and (drop := round(s_champ[src] - s_cand.get(src, 0.0), 9)) > max_drop:
            reasons.append(f"{src} drops {drop * 100:.1f} points (limit {max_drop * 100:g})")
    return {"passed": not reasons, "reasons": reasons, "only_candidate": wins, "only_champion": losses,
            "p": p, "candidate": s_cand, "champion": s_champ}


def champion_samples(tracking_uri: str, model: str = "finance-qa") -> str:
    """The saved answers of the registry's current champion (its `samples` tag)."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.MlflowClient().get_model_version_by_alias(model, "champion").tags["samples"]


def main():
    ap = argparse.ArgumentParser(description="May this candidate replace the champion? "
                                             "Exit 0 = pass, 1 = reject, 2 = error (no decision).")
    ap.add_argument("--candidate", required=True, help="the candidate's saved test answers (samples.jsonl)")
    ap.add_argument("--champion", help="the champion's saved answers (default: the registry's champion)")
    ap.add_argument("--tracking-uri", default="sqlite:///training/qlora_financial/outputs/mlflow.db")
    args = ap.parse_args()
    try:
        d = decide(load_rows(args.candidate), load_rows(args.champion or champion_samples(args.tracking_uri)))
    except Exception as e:  # a crash must not look like "reject" (1) to a pipeline
        print(f"ERROR, no decision: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"{'':16} {'candidate':>9} {'champion':>9}")
    for src in d["champion"]:
        if src != "n":
            print(f"{src:16} {d['candidate'].get(src, 0) * 100:9.1f} {d['champion'][src] * 100:9.1f}")
    print(f"right only with candidate {d['only_candidate']}, only with champion {d['only_champion']}, p = {d['p']:.2g}")
    print("PASS" if d["passed"] else "REJECT: " + "; ".join(d["reasons"]))
    sys.exit(0 if d["passed"] else 1)


if __name__ == "__main__":
    main()
