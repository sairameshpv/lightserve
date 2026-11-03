"""Tests for benchmarks/ab/ab_stats.py on made-up timings: the A/B tool must say "not sure" for
identical sides (and do so honestly: few false alarms), catch a clear 10% slowdown, and never
give a verdict it can't back (one round a side)."""
import random

from benchmarks.ab.ab_run import keep_window
from benchmarks.ab.ab_stats import format_table, percentile, report, round_order, verdict, welch, window_tokens_per_s


def test_identical_values_change_zero_range_contains_zero():
    r = welch([100, 102, 98], [100, 102, 98])
    assert r["change_pct"] == 0
    assert r["lo"] < 0 < r["hi"]


def test_clear_ten_percent_shift_is_detected():
    r = welch([100, 101, 99], [110, 111, 109])
    assert 9 < r["change_pct"] < 11
    assert r["lo"] > 0


def test_false_alarms_on_identical_sides_stay_under_five_percent():
    """The reason for the redesign: a bootstrap over 3 rounds a side called identical sides
    different 22% of the time. Rounds as the samples + a t-interval keeps it under 5%."""
    rng = random.Random(0)
    alarms = 0
    for _ in range(2000):
        r = welch([rng.gauss(100, 3) for _ in range(3)], [rng.gauss(100, 3) for _ in range(3)])
        alarms += not (r["lo"] <= 0 <= r["hi"])
    assert alarms / 2000 <= 0.05


def test_round_order_cancels_steady_drift():
    assert round_order(3) == list("ABBAAB")
    assert round_order(2) == list("ABBA")


def test_percentile_nearest_rank():
    assert percentile(list(range(1, 101)), 50) == 50
    assert percentile(list(range(1, 101)), 99) == 99
    assert percentile([3.0], 99) == 3.0


def test_verdict_depends_on_direction():
    up = {"change_pct": 10, "lo": 4, "hi": 16}
    down = {"change_pct": -10, "lo": -16, "hi": -4}
    unsure = {"change_pct": 2, "lo": -3, "hi": 7}
    assert verdict(up, lower_is_better=True) == "worse"
    assert verdict(down, lower_is_better=True) == "better"
    assert verdict(up, lower_is_better=False) == "better"
    assert verdict(unsure, lower_is_better=True) == "not sure"


def fake_samples(side, rounds, slow, seed, failures=0, n=150, round_noise=0.01):
    """Closed-loop requests: each starts when the last ends; `slow` stretches every time, and
    each round gets its own small shift (a round's shared conditions)."""
    rng = random.Random(seed)
    out = []
    for rnd in rounds:
        t, level = rnd * 1000.0, slow * rng.gauss(1.0, round_noise)
        for i in range(n):
            ttft = 0.05 * level * rng.lognormvariate(0, 0.2)
            total = ttft + 0.5 * level * rng.lognormvariate(0, 0.2)
            ok = i >= failures
            out.append({"side": side, "round": rnd, "t_start": t, "ttft_s": ttft, "total_s": total,
                        "tokens": 32 if ok else 0, "ok": ok})
            t += total
    return out


MEASURES = ("latency p50 (s)", "latency p99 (s)", "TTFT p50 (s)", "TTFT p99 (s)",
            "time/token p50 (ms)", "time/token p99 (ms)", "tokens/s")


def test_report_catches_ten_percent_slower():
    """p50s, time per token and tokens/s catch it. A single round's p99 rests on its slowest ~10
    requests, so with 3 rounds a side a p99 may honestly say "not sure" -- but never "better"."""
    samples = fake_samples("A", [0, 3, 4], 1.0, seed=1, n=1000) + fake_samples("B", [1, 2, 5], 1.1, seed=2, n=1000)
    rows = {r["measure"]: r for r in report(samples)}
    for m in MEASURES:
        assert rows[m]["verdict"] in (("worse", "not sure") if "p99" in m else ("worse",)), (m, rows[m])
    assert -12 < rows["tokens/s"]["change_pct"] < -6


def test_report_says_not_sure_for_identical_sides():
    samples = fake_samples("A", [0, 3, 4], 1.0, seed=1) + fake_samples("B", [1, 2, 5], 1.0, seed=2)
    rows = [r for r in report(samples) if "verdict" in r]
    assert {r["verdict"] for r in rows} == {"not sure"}


def test_report_counts_failures_and_skips_them_in_timings():
    samples = fake_samples("A", [0, 3, 4], 1.0, seed=1) + fake_samples("B", [1, 2, 5], 1.0, seed=1, failures=2)
    rows = {r["measure"]: r for r in report(samples)}
    assert (rows["failures"]["a"], rows["failures"]["b"]) == (0, 6)
    assert rows["latency p50 (s)"]["verdict"] == "not sure"


def test_one_round_a_side_never_gets_a_verdict():
    """One round a side: no round-to-round spread to judge by."""
    samples = fake_samples("A", [0], 1.0, seed=1) + fake_samples("B", [1], 1.01, seed=2)
    rows = [r for r in report(samples) if "verdict" in r]
    assert {r["verdict"] for r in rows} == {"too few rounds"}


def test_spread_shows_rounds_that_disagree():
    """A side whose rounds differ a lot (the Mac slowing mid-run) shows it in the spread column."""
    calm = fake_samples("A", [0, 3, 4], 1.0, seed=1, round_noise=0.0)
    jumpy = fake_samples("B", [1, 2], 1.0, seed=2, round_noise=0.0) + fake_samples("B", [5], 1.2, seed=3, round_noise=0.0)
    rows = {r["measure"]: r for r in report(calm + jumpy)}
    assert rows["time/token p50 (ms)"]["spread_pct"] > 15


def test_format_table_has_a_line_per_measure_and_per_round():
    samples = fake_samples("A", [0, 3], 1.0, seed=1) + fake_samples("B", [1, 2], 1.0, seed=2)
    text = format_table(report(samples))
    assert "time/token p99 (ms)" in text and "not sure" in text
    assert [line.split()[:2] for line in text.splitlines() if line.startswith("round ")] == [
        ["round", "0"], ["round", "1"], ["round", "2"], ["round", "3"]]
    assert "round 1  B" in text


def test_keep_window_drops_warmup_and_late_starts_and_tags():
    t0 = 1000.0
    rows = [{"t_start": t0 + s, "total_s": 1.0} for s in (5, 29.9, 30, 100, 150, 150.1)]
    kept = keep_window(rows, t0, warmup_s=30, round_s=120, side="B", rnd=4)
    assert [r["t_start"] - t0 for r in kept] == [30, 100, 150]
    assert all(r["side"] == "B" and r["round"] == 4 for r in kept)


def test_window_tokens_per_s_counts_only_tokens_made_inside_the_window():
    """A request's tokens spread evenly over its own time; only the part inside the window
    counts. (Dividing by "first start to last end" let the tail -- users finishing their last
    request one by one -- blur a 10% slowdown into "-3%, not sure" on the Mac.)"""
    raw = [{"t_start": 0, "total_s": 10, "tokens": 100, "ok": True},   # half inside: 50
           {"t_start": 6, "total_s": 4, "tokens": 40, "ok": True},     # all inside: 40
           {"t_start": 12, "total_s": 6, "tokens": 60, "ok": True},    # half inside: 30
           {"t_start": 7, "total_s": 1, "tokens": 0, "ok": False}]     # failed: 0
    assert window_tokens_per_s(raw, lo=5, hi=15) == (50 + 40 + 30) / 10


def test_report_uses_the_runners_tokens_per_s_when_given():
    samples = fake_samples("A", [0, 3, 4], 1.0, seed=1) + fake_samples("B", [1, 2, 5], 1.0, seed=2)
    tps = {0: 400, 3: 402, 4: 398, 1: 360, 2: 362, 5: 358}
    rows = {r["measure"]: r for r in report(samples, tokens_per_s=tps)}
    assert rows["tokens/s"]["verdict"] == "worse"
    assert round(rows["tokens/s"]["change_pct"]) == -10


def test_primary_measure_decides_and_leads_the_table():
    """Seven measures = seven chances of a fluke (~1 in 6 identical comparisons shows a false
    verdict somewhere), so one measure, named before the run, decides; the rest are background."""
    samples = fake_samples("A", [0, 3, 4], 1.0, seed=1, n=1000) + fake_samples("B", [1, 2, 5], 1.1, seed=2, n=1000)
    rows = report(samples, primary="latency p99 (s)")
    assert [r["measure"] for r in rows if r.get("primary")] == ["latency p99 (s)"]
    first = format_table(rows).splitlines()[0]
    assert first.startswith("Decision: latency p99 (s) -> worse")


def test_unknown_primary_is_refused():
    samples = fake_samples("A", [0, 3], 1.0, seed=1) + fake_samples("B", [1, 2], 1.0, seed=2)
    try:
        report(samples, primary="latency p95 (s)")
    except ValueError as e:
        assert "time/token p50 (ms)" in str(e)  # lists the valid names
    else:
        raise AssertionError("an unknown primary measure must be refused")
