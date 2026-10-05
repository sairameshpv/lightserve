import json

import huggingface_hub

from training.corpus_prep.prepare_corpus import clean, is_exact_copy, iter_filings, quality_problem

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
