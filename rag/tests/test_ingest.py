"""rag/ingest.py: 10-K text -> tagged, overlapping chunks; the SEC's ticker list -> company names.
Pure Python (no Qdrant, no embedding model)."""
import json

from rag.ingest import chunk_words, filing_chunks, load_names

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
           "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
           "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
           "3": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."}}   # two share classes, one company


def test_chunks_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_words(" ".join(words), size=400, overlap=50)
    assert [len(c.split()) for c in chunks] == [400, 400, 300]
    assert chunks[1].split()[:50] == chunks[0].split()[-50:]          # 50 words shared with the one before
    assert set(" ".join(chunks).split()) == set(words)               # nothing lost


def test_short_and_empty_text():
    assert chunk_words("just a few words", size=400, overlap=50) == ["just a few words"]
    assert chunk_words("   ", size=400, overlap=50) == []


def test_names_by_sec_id_keep_the_first_ticker(tmp_path):
    p = tmp_path / "company_tickers.json"
    p.write_text(json.dumps(TICKERS))
    names = load_names(p)
    assert names[320193] == {"company": "Apple Inc.", "ticker": "AAPL"}
    assert names[1652044]["ticker"] == "GOOGL"


def test_a_filing_becomes_chunks_tagged_with_company_year_and_section():
    names = {320193: {"company": "Apple Inc.", "ticker": "AAPL"}}
    filing = {"cik": "0000320193", "year": "2019", "filename": "320193_2019.txt",
              "section_1": "Business text that is not indexed.",
              "section_7": " ".join(["revenue"] * 450),               # 2 chunks
              "section_8": "Total net sales 260,174 265,595",
              "section_7A": ""}
    chunks = filing_chunks(filing, names, size=400, overlap=50)
    assert [(c["section"], c["chunk"]) for c in chunks] == [("7", 0), ("7", 1), ("8", 0)]   # Items 7 and 8 only
    first = chunks[0]
    assert {k: first[k] for k in ("cik", "company", "ticker", "year")} == \
        {"cik": 320193, "company": "Apple Inc.", "ticker": "AAPL", "year": 2019}
    assert chunks[2]["text"] == "Total net sales 260,174 265,595"


def test_a_company_not_on_the_ticker_list_keeps_its_sec_id():
    chunks = filing_chunks({"cik": "99", "year": "2019", "section_7": "text", "section_8": ""}, {})
    assert (chunks[0]["company"], chunks[0]["ticker"]) == ("CIK 99", None)


# --- the streaming build (fake embedder + fake Qdrant: no model, no server) ---

import io

from rag.ingest import build, iter_filings


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


class FakeQdrant:
    def __init__(self):
        self.points = []

    def upsert(self, name, points):
        self.points += points


def jsonl(*rows):
    return io.BytesIO("".join(json.dumps(r) + "\n" for r in rows).encode())


def test_iter_filings_reads_one_filing_per_line_and_skips_blank_lines():
    stream = io.BytesIO(b'{"cik": "1", "year": "2019"}\n\n{"cik": "2", "year": "2019"}\n')
    assert [f["cik"] for f in iter_filings(stream)] == ["1", "2"]


def test_build_streams_every_filing_into_qdrant_with_unique_ids():
    names = {320193: {"company": "Apple Inc.", "ticker": "AAPL"}}
    streams = [jsonl({"cik": "320193", "year": "2019", "section_7": "a " * 500, "section_8": "b"}),
               jsonl({"cik": "99", "year": "2019", "section_7": "c", "section_8": ""})]   # a second split file
    qc, emb = FakeQdrant(), FakeEmbedder()
    counts = build(streams, names, qc, emb, batch=2)
    assert counts == {"filings": 2, "chunks": 4}
    assert sorted(p.id for p in qc.points) == [0, 1, 2, 3]
    assert {p.payload["company"] for p in qc.points} == {"Apple Inc.", "CIK 99"}


def test_build_stops_after_a_limit_for_the_sample_run():
    rows = [{"cik": str(i), "year": "2019", "section_7": "x", "section_8": ""} for i in range(10)]
    qc = FakeQdrant()
    assert build([jsonl(*rows)], {}, qc, FakeEmbedder(), limit=3)["filings"] == 3
