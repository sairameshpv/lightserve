"""Correctness tests for benchmarks/prefix_caching/measure_ttft.py's
torch-free helpers (build_workload, summarize_results) -- the only parts of
that script verifiable without a real CUDA GPU (see its module docstring).
Pure Python, no torch/CUDA, runs anywhere.
"""
import pytest

from benchmarks.prefix_caching.measure_ttft import build_workload, summarize_results


class TestBuildWorkload:
    def test_returns_one_prompt_per_request(self):
        prompts = build_workload(prefix_len=8, num_requests=3, suffix_len=2, vocab_size=100)
        assert len(prompts) == 3

    def test_every_prompt_has_prefix_plus_suffix_length(self):
        prompts = build_workload(prefix_len=8, num_requests=3, suffix_len=2, vocab_size=100)
        assert all(len(p) == 10 for p in prompts)

    def test_every_prompt_shares_the_same_prefix(self):
        prompts = build_workload(prefix_len=8, num_requests=5, suffix_len=2, vocab_size=100)
        prefixes = {tuple(p[:8]) for p in prompts}
        assert len(prefixes) == 1  # all identical

    def test_suffixes_are_not_all_identical(self):
        prompts = build_workload(prefix_len=8, num_requests=5, suffix_len=4, vocab_size=100)
        suffixes = {tuple(p[8:]) for p in prompts}
        assert len(suffixes) > 1  # random per request, vanishingly unlikely to collide

    def test_zero_prefix_len_is_just_the_suffix(self):
        prompts = build_workload(prefix_len=0, num_requests=2, suffix_len=4, vocab_size=100)
        assert all(len(p) == 4 for p in prompts)

    def test_same_seed_is_reproducible(self):
        a = build_workload(prefix_len=8, num_requests=3, suffix_len=2, vocab_size=100, seed=7)
        b = build_workload(prefix_len=8, num_requests=3, suffix_len=2, vocab_size=100, seed=7)
        assert a == b


class TestSummarizeResults:
    def test_reduction_pct_arithmetic(self):
        ttft_off = {"r0": 0.100, "r1": 0.100}  # mean 100ms
        ttft_on = {"r0": 0.040, "r1": 0.060}   # mean 50ms -- 50% reduction
        result = summarize_results(prefix_len=512, ttft_off=ttft_off, ttft_on=ttft_on)
        assert result["prefix_len"] == 512
        assert result["cache_off_ttft_ms"] == 100.0
        assert result["cache_on_ttft_ms"] == 50.0
        assert result["reduction_pct"] == 50.0

    def test_no_reduction_when_ttft_unchanged(self):
        ttft = {"r0": 0.080}
        result = summarize_results(prefix_len=0, ttft_off=ttft, ttft_on=dict(ttft))
        assert result["reduction_pct"] == 0.0

    def test_negative_reduction_when_cache_on_is_slower(self):
        ttft_off = {"r0": 0.050}
        ttft_on = {"r0": 0.075}
        result = summarize_results(prefix_len=0, ttft_off=ttft_off, ttft_on=ttft_on)
        assert result["reduction_pct"] == pytest.approx(-50.0)