"""make_traffic.py: synthetic query windows with a known change, from made-up rows. No GPU."""
import pytest

from training.qlora_financial.make_traffic import window


def rows():
    """100 short TAT-QA questions and 100 FinQA ones whose documents are 1..100 words long."""
    tat = [{"source": "TAT-QA", "messages": [{}, {"content": f"a\tb\nQuestion: t{i}?"}]} for i in range(100)]
    fin = [{"source": "FinQA", "messages": [{}, {"content": "word " * (i + 1) + f"\nQuestion: f{i}?"}]} for i in range(100)]
    return tat + fin


def test_none_is_a_plain_random_sample():
    w = window(rows(), "none", n=60, seed=0)
    assert len(w) == 60 and set(w[0]) == {"prompt", "source"}
    assert len({q["prompt"] for q in w}) == 60  # no question twice


def test_source_is_80_percent_tat_qa():
    w = window(rows(), "source", n=60, seed=0)
    assert sum(q["source"] == "TAT-QA" for q in w) == 48  # 80% of 60


def test_length_takes_only_the_longest_documents():
    w = window([r for r in rows() if r["source"] == "FinQA"], "length", n=20, seed=0)
    assert min(q["prompt"].count("word") for q in w) >= 67  # the longest third of 1..100 words


def test_mix_puts_part_of_the_change_into_normal_traffic():
    w = window(rows(), "source", n=60, seed=0, mix=0.5)  # 30 drifted (24 TAT-QA) + 30 random
    assert len(w) == 60 and len({q["prompt"] for q in w}) == 60
    assert 24 <= sum(q["source"] == "TAT-QA" for q in w) <= 54


def test_same_seed_same_window_and_unknown_kind_refused():
    assert window(rows(), "source", n=60, seed=3) == window(rows(), "source", n=60, seed=3)
    assert window(rows(), "source", n=60, seed=3) != window(rows(), "source", n=60, seed=4)
    with pytest.raises(ValueError):
        window(rows(), "weekend", n=60)


def test_mix_zero_is_plain_random_traffic():
    assert window(rows(), "source", n=60, seed=5, mix=0.0) == window(rows(), "none", n=60, seed=5)
