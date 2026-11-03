"""A/B test one setting on a running server: the same Locust workload in turns (A B B A A B),
every request's own timings kept, then B's change vs A per measure with a 95% range and a
verdict (benchmarks/ab/ab_stats.py). See benchmarks/ab/README.md.

    python benchmarks/ab/ab_run.py --host http://localhost:8000 --out runs/persistence \\
        --a-setup "sudo nvidia-smi -pm 0" --b-setup "sudo nvidia-smi -pm 1"

Each round: that side's setup command (a shell command: restart the server with a flag, change
an OS setting...), wait for /health, then Locust for --warmup-s + --round-s. Only requests that
start inside the measured window count; Locust lets in-flight requests finish (--stop-timeout),
so long requests aren't cut off at the end (that would favour short ones).
Writes <out>/samples.jsonl (every kept request, tagged side + round), tokens_per_s.json (per
round, tokens made inside the window), report.txt, report.json; round<n>_<side>.jsonl keeps every
request of a round, warm-up included.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.ab.ab_stats import MEASURES, PRIMARY, format_table, report, round_order, window_tokens_per_s  # noqa: E402

LOCUSTFILE = Path(__file__).resolve().parents[1] / "locustfile.py"


def keep_window(rows, t0, warmup_s, round_s, side, rnd):
    """Requests that started in [t0 + warmup, t0 + warmup + round], tagged with side and round."""
    lo, hi = t0 + warmup_s, t0 + warmup_s + round_s
    return [{**r, "side": side, "round": rnd} for r in rows if lo <= r["t_start"] <= hi]


def wait_healthy(host, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"{host}/health", timeout=5).status == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit(f"{host}/health not OK after {timeout_s} s")


def run_round(args, side, rnd, out):
    setup = args.a_setup if side == "A" else args.b_setup
    if setup:
        subprocess.run(setup, shell=True, check=True)
    wait_healthy(args.host, args.health_timeout_s)
    raw = out / f"round{rnd}_{side}.jsonl"
    t0 = time.time()
    with (out / f"round{rnd}_{side}_locust.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--host", args.host, "--headless",
                        "--users", str(args.users), "--spawn-rate", str(args.users),
                        "--run-time", f"{args.warmup_s + args.round_s}s", "--stop-timeout", str(args.stop_timeout_s),
                        "--target", args.target, "--model", args.model, "--stream", "--samples-out", str(raw),
                        "--seed", str(args.seed), "--only-summary"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    rows = [json.loads(line) for line in raw.open()]
    lo = t0 + args.warmup_s
    return keep_window(rows, t0, args.warmup_s, args.round_s, side, rnd), window_tokens_per_s(rows, lo, lo + args.round_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True, help="folder for samples and the report (must not exist)")
    ap.add_argument("--a-setup", help="shell command run before each A round")
    ap.add_argument("--b-setup", help="shell command run before each B round")
    ap.add_argument("--rounds-per-side", type=int, default=3)
    ap.add_argument("--warmup-s", type=int, default=30)
    ap.add_argument("--round-s", type=int, default=120)
    ap.add_argument("--stop-timeout-s", type=int, default=120)
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--target", default="lightserve", choices=["lightserve", "vllm"])
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--health-timeout-s", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0, help="same prompts in the same order every round")
    ap.add_argument("--primary", default=PRIMARY, choices=[m[0] for m in MEASURES],
                    help="the one measure whose verdict decides; name it before the run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    samples, tokens_per_s = [], {}
    for rnd, side in enumerate(round_order(args.rounds_per_side)):
        kept, tokens_per_s[rnd] = run_round(args, side, rnd, out)
        print(f"round {rnd} {side}: {len(kept)} requests, {sum(not r['ok'] for r in kept)} failed", flush=True)
        samples += kept
    with (out / "samples.jsonl").open("w") as f:
        f.writelines(json.dumps(s) + "\n" for s in samples)
    (out / "tokens_per_s.json").write_text(json.dumps(tokens_per_s))
    rows = report(samples, tokens_per_s, args.primary)
    (out / "report.json").write_text(json.dumps(rows, indent=1))
    (out / "report.txt").write_text(format_table(rows) + "\n")
    print(format_table(rows))


if __name__ == "__main__":
    main()
