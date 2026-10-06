import json

import huggingface_hub

import random

from training.corpus_prep.prepare_corpus import (NearCopyIndex, clean, is_exact_copy, iter_filings, minhash,
                                                 quality_problem)


def _words(n, seed):
    rng = random.Random(seed)
    return [f"w{rng.randrange(5000)}" for _ in range(n)]


def _edit(words, share, seed):  # replace a share of the words, like changed numbers and names
    rng = random.Random(seed)
    return [f"new{rng.randrange(10**6)}" if rng.random() < share else w for w in words]


def _jaccard(a, b):  # true similarity of the 5-word shingle sets
    sa, sb = ({tuple(x[i:i + 5]) for i in range(len(x) - 4)} for x in (a, b))
    return len(sa & sb) / len(sa | sb)


def test_minhash_estimates_true_similarity():
    base = _words(3000, seed=1)
    for share in (0.01, 0.03, 0.1):
        other = _edit(base, share, seed=2)
        estimate = float((minhash(" ".join(base)) == minhash(" ".join(other))).mean())
        assert abs(estimate - _jaccard(base, other)) < 0.1  # 128 values: typical error ~0.03-0.04


def test_near_copy_index_catches_light_edits_only():
    base = _words(3000, seed=1)
    index = NearCopyIndex()
    assert index.check_and_add(minhash(" ".join(base))) is None                    # first: kept
    assert index.check_and_add(minhash(" ".join(_edit(base, 0.01, 3))))[0] == 0    # 1% of words changed (~93% similar): near-copy of #0
    assert index.check_and_add(minhash(" ".join(_words(3000, seed=4)))) is None    # unrelated: kept

REPORT = "\n".join(f"In {2000 + i} revenue in segment {i} grew because demand was strong." for i in range(60))


def test_quality_passes_a_normal_report_even_with_page_headers():
    # Like real 10-Ks: a "•" on its own line before each point, a header once per 10 points ("page").
    # About half the lines are repeats (the old per-line rule failed it), but only ~7% of the characters.
    lines = [f"•\n{ln}" for ln in REPORT.split("\n")]
    with_headers = "\n".join("Notes to Consolidated Financial Statements\n" + "\n".join(lines[i:i + 10])
                             for i in range(0, len(lines), 10))
    assert quality_problem(REPORT) is None
    assert quality_problem(with_headers) is None  # short repeated headers: the fix from stage B


def test_quality_rejects_each_kind_of_junk():
    paragraph = " ".join(f"The company recorded item {i} under the standard." for i in range(20))
    assert quality_problem("Item 1. Business") == "too_few_words"
    assert quality_problem("x " * 300) == "odd_word_length"
    assert quality_problem(" ".join(str(1000 + i) for i in range(300))) == "few_alpha_words"
    assert quality_problem("\n".join([paragraph] * 4)) == "repeated_lines"  # pasted 4 times


def test_exact_copy_caught_only_for_identical_text():
    seen = set()
    assert [is_exact_copy(t, seen) for t in (REPORT, REPORT + ".", REPORT)] == [False, False, True]


def test_clean_drops_page_numbers_but_keeps_years():
    text = "Item 1. Business\n12\n- 13 -\nPage 14\nF-3\nTable of Contents\nRevenue grew.\n2019"
    assert clean(text) == "Item 1. Business\nRevenue grew.\n2019"


def test_clean_fixes_odd_spaces_ligatures_and_blank_lines():
    assert clean("The ﬁrm   earned\n\n\n\nmore.") == "The firm earned\n\nmore."


def test_iter_filings_joins_nonempty_sections_in_order(tmp_path, monkeypatch):
    rows = [{"filename": "1_2019.htm", "cik": "1", "year": "2019",
             "section_7": "Item 7. MD&A", "section_1": "Item 1. Business ", "section_1A": "  "},
            {"filename": "2_2019.htm", "cik": "2", "year": "2019", "section_1": "x"}]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(path))  # no network
    docs = list(iter_filings(2019, "train", max_docs=1))
    assert docs == [{"id": "1_2019.htm", "cik": "1", "year": 2019, "text": "Item 1. Business\n\nItem 7. MD&A"}]
