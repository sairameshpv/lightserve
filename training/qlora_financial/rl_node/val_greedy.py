"""Greedy accuracy on the 299 val questions, scored like OpenRLHF eval
(rl_reward.score, 512 tokens). With a reference json (v3's), also prints
answers changed, wins/losses and a two-sided sign-test p."""
import json, math, sys
from concurrent.futures import ThreadPoolExecutor
from training.qlora_financial.rl_reward import score
from training.qlora_financial.sample_eval import chat

rows = [json.loads(l) for l in open("/rl/data/val.jsonl")]
with ThreadPoolExecutor(32) as pool:
    answers = list(pool.map(lambda r: chat("http://localhost:8000", r["prompt"], 1, 0.0, 512)[0], rows))
ok = [score(a, r["label"]) for a, r in zip(answers, rows)]
json.dump([{"answer": a, "ok": o} for a, o in zip(answers, ok)], open(sys.argv[1], "w"))
stats = {"acc": round(sum(ok) / len(ok), 4), "right": int(sum(ok)), "steps": round(sum("Step " in a for a in answers) / len(answers), 4)}
if len(sys.argv) > 2:
    ref = json.load(open(sys.argv[2]))
    wins = int(sum(o and not r["ok"] for o, r in zip(ok, ref)))
    losses = int(sum(r["ok"] and not o for o, r in zip(ok, ref)))
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    stats.update(changed=sum(a != r["answer"] for a, r in zip(answers, ref)), wins=wins, losses=losses, p=round(p, 3))
print(json.dumps(stats))
