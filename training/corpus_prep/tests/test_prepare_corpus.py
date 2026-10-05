import json

import huggingface_hub

from training.corpus_prep.prepare_corpus import clean, iter_filings


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
