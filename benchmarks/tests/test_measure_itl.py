"""Correctness tests for benchmarks/chunked_prefill/measure_itl.py's
torch-free helpers (build_decode_prompts, build_prefill_prompt,
summarize_itl, aggregate_repeats) -- the only parts of that script
verifiable without a real CUDA GPU (see its module docstring). Pure
Python, no torch/CUDA, runs anywhere.
"""
import statistics

import pytest

from benchmarks.chunked_prefill.measure_itl import (
    aggregate_repeats,
    build_decode_prompts,
    build_prefill_prompt,
    summarize_itl,
)


class TestBuildDecodePrompts:
    def test_returns_one_prompt_per_request(self):
        prompts = build_decode_prompts(4, 8, vocab_size=100)
        assert len(prompts) == 4

    def test_every_prompt_has_the_requested_length(self):
        prompts = build_decode_prompts(4, 8, vocab_size=100)
        assert all(len(p) == 8 for p in prompts)

    def test_prompts_are_not_all_identical(self):
        prompts = build_decode_prompts(5, 8, vocab_size=100)
        assert len({tuple(p) for p in prompts}) > 1  # independent draws, vanishingly unlikely to collide

    def test_same_seed_is_reproducible(self):
        a = build_decode_prompts(4, 8, vocab_size=100, seed=7)
        b = build_decode_prompts(4, 8, vocab_size=100, seed=7)
        assert a == b


class TestBuildPrefillPrompt:
    def test_returns_the_requested_length(self):
        assert len(build_prefill_prompt(256, vocab_size=100)) == 256

    def test_same_seed_is_reproducible(self):
        a = build_prefill_prompt(64, vocab_size=100, seed=3)
        b = build_prefill_prompt(64, vocab_size=100, seed=3)
        assert a == b

    def test_default_seed_differs_from_decode_prompts_default(self):
        # Different default seeds (0 vs 1) so a real run's decode and
        # prefill workloads never accidentally draw identical content.
        decode = build_decode_prompts(1, 32, vocab_size=100)[0]
        prefill = build_prefill_prompt(32, vocab_size=100)
        assert decode != prefill


class TestSummarizeItl:
    def test_separates_baseline_from_disrupted(self):
        records = [
            {"phase": "baseline", "itl_seconds": 0.010},
            {"phase": "baseline", "itl_seconds": 0.012},
            {"phase": "disrupted", "itl_seconds": 0.050},
            {"phase": "disrupted", "itl_seconds": 0.400},
        ]
        result = summarize_itl(chunk_size=512, itl_records=records, prefill_ttft=0.600)
        assert result["chunk_size"] == 512
        assert result["baseline_itl_ms_mean"] == pytest.approx(11.0)
        assert result["disrupted_itl_ms_mean"] == pytest.approx(225.0)
        assert result["disrupted_itl_ms_max"] == pytest.approx(400.0)
        assert result["prefill_ttft_ms"] == pytest.approx(600.0)

    def test_empty_phase_reports_zero_not_an_error(self):
        records = [{"phase": "baseline", "itl_seconds": 0.010}]
        result = summarize_itl(chunk_size=512, itl_records=records, prefill_ttft=None)
        assert result["disrupted_itl_ms_mean"] == 0.0
        assert result["disrupted_itl_ms_max"] == 0.0
        assert result["prefill_ttft_ms"] == 0.0  # prefill_ttft=None (never got a token) -- not a crash


class TestAggregateRepeats:
    def _summary(self, baseline, disrupted_mean, disrupted_max, ttft):
        return {
            "baseline_itl_ms_mean": baseline,
            "disrupted_itl_ms_mean": disrupted_mean,
            "disrupted_itl_ms_max": disrupted_max,
            "prefill_ttft_ms": ttft,
        }

    def test_single_repeat_has_zero_stdev(self):
        result = aggregate_repeats(1024, [self._summary(10.0, 50.0, 200.0, 600.0)])
        assert result["num_repeats"] == 1
        assert result["disrupted_itl_ms_mean"] == 50.0
        assert result["disrupted_itl_ms_stdev"] == 0.0

    def test_multiple_repeats_average_and_report_spread(self):
        repeats = [
            self._summary(10.0, 50.0, 200.0, 600.0),
            self._summary(12.0, 60.0, 250.0, 620.0),
            self._summary(8.0, 40.0, 150.0, 580.0),
        ]
        result = aggregate_repeats(1024, repeats)
        assert result["num_repeats"] == 3
        assert result["disrupted_itl_ms_mean"] == pytest.approx(50.0)
        assert result["disrupted_itl_ms_stdev"] == pytest.approx(statistics.stdev([50.0, 60.0, 40.0]))

    def test_chunk_size_passes_through_unchanged(self):
        result = aggregate_repeats(8192, [self._summary(1.0, 1.0, 1.0, 1.0)])
        assert result["chunk_size"] == 8192
