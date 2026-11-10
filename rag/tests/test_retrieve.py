import json
import zlib

import pytest

from rag.ingest import add_chunks, make_collection
from rag.retrieve import find_company, find_year, load_lookup, search

APPLE, AMAZON, ALPHABET, JNJ, APPLE_REIT = 320193, 1018724, 1652044, 200406, 1418121
FORD, XOM, USB, FUND = 37996, 34088, 36104, 1000001


@pytest.fixture
def lookup(tmp_path):
    rows = [(ALPHABET, "GOOGL", "Alphabet Inc."), (APPLE, "AAPL", "Apple Inc."),
            (AMAZON, "AMZN", "AMAZON COM INC"), (ALPHABET, "GOOG", "Alphabet Inc."),
            (JNJ, "JNJ", "JOHNSON & JOHNSON"), (APPLE_REIT, "APLE", "Apple Hospitality REIT, Inc."),
            (FORD, "F", "FORD MOTOR CO"), (XOM, "XOM", "ExxonMobil Holdings Corp"), (USB, "USB", "US BANCORP \\DE\\"),
            (FUND, "TRSF", "Total Return Securities Fund")]
    path = tmp_path / "tickers.json"
    path.write_text(json.dumps({str(i): {"cik_str": c, "ticker": t, "title": n} for i, (c, t, n) in enumerate(rows)}))
    return load_lookup(path)


@pytest.mark.parametrize("question, cik", [
    ("What was Apple's revenue growth in 2019?", APPLE),
    ("AAPL revenue in 2019", APPLE),
    ("How did Amazon's net sales change?", AMAZON),       # SEC title "AMAZON COM INC"
    ("GOOG operating income", ALPHABET),                  # any of a company's tickers
    ("GOOGL operating income", ALPHABET),
    ("Johnson & Johnson R&D spending", JNJ),
    ("Apple Hospitality REIT occupancy", APPLE_REIT),     # the longest name wins
    ("How much debt did Ford have?", FORD),               # a name's first word ("FORD MOTOR CO")
    ("Exxon Mobil capital spending", XOM),                # spaces ignored ("ExxonMobil")
])
def test_find_company(lookup, question, cik):
    assert find_company(question, lookup) == cik


@pytest.mark.parametrize("question", ["What was revenue growth in 2019?",
                                      "What was the US GDP growth?",    # "US" is too short to be a first word
                                      "Total revenue growth in 2019?",  # a question's own first word is not a name's
                                      "Ford's debt in 2019?"])          # ...so a nickname can't start the question
def test_find_company_none(lookup, question):
    assert find_company(question, lookup) is None


@pytest.mark.parametrize("question, year", [
    ("revenue growth in 2019?", 2019),
    ("2019 compared to 2018", 2019),   # the later year: its 10-K holds both
    ("FY2019 margin", 2019),
    ("revenue growth?", None),
])
def test_find_year(question, year):
    assert find_year(question) == year


class FakeEmbedder:
    """Each word lands in one of 384 slots: texts sharing words point the same way."""
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 384
            for w in t.lower().split():
                v[zlib.crc32(w.encode()) % 384] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture
def index():
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    make_collection(client, "t")
    chunks = [{"cik": c, "company": n, "ticker": None, "year": y, "section": "7", "chunk": 0,
               "text": f"{n} total net sales revenue in {y}"}
              for c, n in ((APPLE, "Apple Inc."), (AMAZON, "AMAZON COM INC")) for y in (2018, 2019)]
    add_chunks(client, FakeEmbedder(), chunks, 0, "t")
    return client


def test_search_filters_company_and_year(index, lookup):
    r = search(index, FakeEmbedder(), "What was Apple's net sales revenue in 2019?", lookup, k=5, collection="t")
    assert (r["cik"], r["company"], r["year"]) == (APPLE, "Apple Inc.", 2019)
    assert len(r["passages"]) == 1
    assert {(p["cik"], p["year"]) for p in r["passages"]} == {(APPLE, 2019)}


def test_search_without_company_or_year_searches_all(index, lookup):
    r = search(index, FakeEmbedder(), "total net sales revenue", lookup, k=5, collection="t")
    assert r["cik"] is None and r["year"] is None
    assert len(r["passages"]) == 4


def test_search_keeps_the_questions_company_and_year(index, lookup):
    """The agent's follow-up search ("net sales") still searches the question's company and year."""
    r = search(index, FakeEmbedder(), "total net sales", lookup, k=5, collection="t", cik=APPLE, year=2019)
    assert {(p["cik"], p["year"]) for p in r["passages"]} == {(APPLE, 2019)}
    assert r["company"] == "Apple Inc."
    # The question's company and year hold even when the text names others (review: "Operating
    # Income" is a company's name, and "2018" isn't in the index).
    r = search(index, FakeEmbedder(), "AMAZON net sales in 2018", lookup, k=5, collection="t", cik=APPLE, year=2019)
    assert {(p["cik"], p["year"]) for p in r["passages"]} == {(APPLE, 2019)}
