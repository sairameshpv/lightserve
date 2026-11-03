"""Is setting B faster or slower than setting A, or can't we tell? The statistics behind
benchmarks/ab/ab_run.py. Each measure is computed once per round (a round's requests share that
moment's conditions, so a round, not a request, is one sample); B's change vs A is then a
Welch t-interval over the rounds: honest with only 3 rounds a side (a bootstrap over 3 rounds
called identical sides different 22% of the time; this stays under 5%), so cautious.
Pure Python, no numpy/scipy.
"""
import math
import statistics

# two-sided 95% t critical values by degrees of freedom; df is rounded down (the cautious side)
_T975 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def welch(a, b):
    """B's change vs A in % (of A's mean), with its 95% range {lo, hi}; a and b: one value per round."""
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return {"change_pct": (mb / ma - 1) * 100, "lo": (mb / ma - 1) * 100, "hi": (mb / ma - 1) * 100}
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = _T975[int(df)] if int(df) < len(_T975) else 1.96
    return {"change_pct": (mb - ma) / ma * 100, "lo": (mb - ma - t * se) / ma * 100, "hi": (mb - ma + t * se) / ma * 100}


def round_order(n_per_side):
    """A B, B A, A B, ...: a steady drift (warming GPU, a neighbour starting) hits both sides
    about equally, which plain A B A B wouldn't."""
    return [s for i in range(n_per_side) for s in ("AB" if i % 2 == 0 else "BA")]


def percentile(values, p):
    """Nearest-rank percentile: an actual measured value, no interpolation."""
    v = sorted(values)
    return v[max(0, -(-len(v) * p // 100) - 1)]


def verdict(r, lower_is_better):
    """"better"/"worse" only when the whole 95% range is on one side of 0."""
    if r["lo"] <= 0 <= r["hi"]:
        return "not sure"
    return "better" if (r["hi"] < 0) == lower_is_better else "worse"


def window_tokens_per_s(raw, lo, hi):
    """Tokens made inside the measured window [lo, hi] per second: each request's tokens spread
    evenly over its own time, only the part inside the window counted. `raw`: every request of the
    round, warm-up included (ab_run.py keeps them in round<n>_<side>.jsonl)."""
    def inside(r):
        return max(0.0, min(hi, r["t_start"] + r["total_s"]) - max(lo, r["t_start"])) / r["total_s"]
    return sum(r["tokens"] * inside(r) for r in raw if r["ok"]) / (hi - lo)


def _tokens_per_s(reqs):
    """Fallback when only the kept samples exist: tokens / the round's span (first start to last
    end). Blurrier than window_tokens_per_s: the span includes a tail of users finishing their
    last request one by one, which varies round to round."""
    span = max(r["t_start"] + r["total_s"] for r in reqs) - min(r["t_start"] for r in reqs)
    return sum(r["tokens"] for r in reqs if r["ok"]) / span


def _per_token_ms(reqs):
    """Time per output token after the first, per request: steady whatever the answer's length."""
    return [1000 * (r["total_s"] - r["ttft_s"]) / (r["tokens"] - 1) for r in reqs if r["ok"] and r["tokens"] > 1]


MEASURES = [  # name, short name, value of one round's requests, lower is better
    ("latency p50 (s)", "lat50", lambda q: percentile([r["total_s"] for r in q if r["ok"]], 50), True),
    ("latency p99 (s)", "lat99", lambda q: percentile([r["total_s"] for r in q if r["ok"]], 99), True),
    ("TTFT p50 (s)", "ttft50", lambda q: percentile([r["ttft_s"] for r in q if r["ok"]], 50), True),
    ("TTFT p99 (s)", "ttft99", lambda q: percentile([r["ttft_s"] for r in q if r["ok"]], 99), True),
    ("time/token p50 (ms)", "tpt50", lambda q: percentile(_per_token_ms(q), 50), True),
    ("time/token p99 (ms)", "tpt99", lambda q: percentile(_per_token_ms(q), 99), True),
    ("tokens/s", "tok/s", _tokens_per_s, False),
]


def _rounds(samples, side):
    by = {}
    for s in samples:
        if s["side"] == side:
            by.setdefault(s["round"], []).append(s)
    return {rnd: by[rnd] for rnd in sorted(by)}


def _spread(values):
    """How far one side's rounds disagree: (max - min) / mean, in %."""
    return (max(values) - min(values)) / statistics.mean(values) * 100


PRIMARY = "time/token p50 (ms)"  # the steadiest measure, and what speed settings change


def report(samples, tokens_per_s=None, primary=PRIMARY):
    """One row per measure: A's and B's mean over rounds, B's change in % with its 95% range,
    the verdict ("too few rounds" with < 2 rounds a side), the widest within-side spread, and
    every round's own value. Last row: failure counts. `tokens_per_s`: {round: value} from
    window_tokens_per_s (the runner's); without it, the blurrier _tokens_per_s fallback.
    `primary`: the one measure, named before the run, whose verdict decides; seven measures are
    seven chances of a fluke (~1 in 6 identical comparisons shows a false verdict somewhere)."""
    names = [m[0] for m in MEASURES]
    if primary not in names:
        raise ValueError(f"unknown primary measure {primary!r}; one of: {', '.join(names)}")
    rounds = {side: _rounds(samples, side) for side in "AB"}
    few = min(len(rounds["A"]), len(rounds["B"])) < 2
    rows = []
    for name, short, value, lower in MEASURES:
        if name == "tokens/s" and tokens_per_s is not None:
            per = {side: {rnd: tokens_per_s[rnd] for rnd in rounds[side]} for side in "AB"}
        else:
            per = {side: {rnd: value(q) for rnd, q in rounds[side].items()} for side in "AB"}
        a, b = list(per["A"].values()), list(per["B"].values())
        row = {"measure": name, "short": short, "primary": name == primary, "a": statistics.mean(a), "b": statistics.mean(b),
               "rounds": {rnd: (side, v) for side in "AB" for rnd, v in per[side].items()}}
        if few:
            row.update(change_pct=(row["b"] / row["a"] - 1) * 100, lo=None, hi=None, verdict="too few rounds", spread_pct=None)
        else:
            r = welch(a, b)
            row.update(r, verdict=verdict(r, lower), spread_pct=max(_spread(a), _spread(b)))
        rows.append(row)
    fails = {side: sum(not s["ok"] for q in rounds[side].values() for s in q) for side in "AB"}
    rows.append({"measure": "failures", "a": fails["A"], "b": fails["B"]})
    return rows


def format_table(rows):
    """The verdict table, then one line per round (so drift between rounds is visible)."""
    decide = next(r for r in rows if r.get("primary"))
    lines = [f"Decision: {decide['measure']} -> {decide['verdict']} (named before the run; the rest is background)", "",
             f"{'measure':<20} {'A':>9} {'B':>9} {'change':>8}  {'95% range':<18} {'spread':>6}  verdict", "-" * 88]
    timed = [r for r in rows if "verdict" in r]
    for r in timed:
        rng = "" if r["lo"] is None else f"{r['lo']:+.1f}% .. {r['hi']:+.1f}%"
        spread = "" if r["spread_pct"] is None else f"{r['spread_pct']:.0f}%"
        mark = "  <- decides" if r["primary"] else ""
        lines.append(f"{r['measure']:<20} {r['a']:>9.3f} {r['b']:>9.3f} {r['change_pct']:>+7.1f}%  {rng:<18} {spread:>6}  {r['verdict']}{mark}")
    for r in rows:
        if "verdict" not in r:
            lines.append(f"{r['measure']:<20} {r['a']:>9} {r['b']:>9}")
    lines += ["", "spread = how far one side's own rounds disagree, (max - min) / mean, the wider side", "",
              f"{'per round':<10} " + "".join(f"{r['short']:>10}" for r in timed)]
    for rnd in sorted(timed[0]["rounds"]):
        side = timed[0]["rounds"][rnd][0]
        lines.append(f"round {rnd}  {side} " + "".join(f"{r['rounds'][rnd][1]:>10.3f}" for r in timed))
    return "\n".join(lines)
