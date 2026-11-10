"""A question -> which company and year it asks about -> the best-matching 10-K passages.

The company is found by name ("Apple", "Johnson & Johnson") or ticker ("AAPL") against the SEC's
list; the year is the fiscal year the filing covers ("2019"). The search only looks inside that
company's and that year's passages, so another company's numbers can't come back.
"""
import json
import re
from pathlib import Path

from rag.ingest import COLLECTION, load_names

# Words that end company names but that people leave out: "AMAZON COM INC" is asked as "Amazon".
SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "com", "company", "ltd", "limited",
            "plc", "llc", "lp", "holdings", "holding", "group", "sa", "nv", "ag", "de", "the"}
# bge models want this before a question (not before the passages) for search.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _words(text: str) -> list:
    """Words and '&', with possessives dropped: "Apple's R&D" -> Apple, R, &, D."""
    return re.findall(r"[A-Za-z0-9]+|&", re.sub(r"['’]s\b", "", text))


def _strip(words: list) -> list:
    words = [w.lower() for w in words]
    while words and (words[-1] in SUFFIXES or words[-1] == "&"):
        words.pop()
    while words and words[0] == "the":
        words.pop(0)
    return words


def _key(words: list) -> str:
    """Spaces ignored, so "Exxon Mobil" and "ExxonMobil" match."""
    return "".join(_strip(words))


def load_lookup(path) -> dict:
    """The SEC list, ready for matching. The file is ordered by company size, so when two companies
    share a name or a ticker, the bigger one wins."""
    rows = list(json.loads(Path(path).read_text()).values())
    lookup = {"names": load_names(path), "tickers": {}, "by_name": {}, "by_first": {}, "rank": {}}
    for i, row in enumerate(rows):
        cik = int(row["cik_str"])
        lookup["rank"].setdefault(cik, i)
        lookup["tickers"].setdefault(row["ticker"].upper(), cik)
        words = _strip(_words(row["title"]))
        if words:
            lookup["by_name"].setdefault("".join(words), cik)
        for n in range(1, len(words)):          # "FORD MOTOR CO" is asked as "Ford"
            first = "".join(words[:n])
            if len(first) >= 4:
                lookup["by_first"].setdefault(first, cik)
    return lookup


def find_company(question: str, lookup: dict, longest: int = 6):
    """The company's SEC ID, or None. A name must start with a capital letter in the question.
    Tried in order: a full name, then a name's first word(s) (4+ letters, not the question's own
    first word: "What was Ford's debt"), then a ticker
    (2+ capitals: "AAPL"). Within a kind, the longest match wins, then the bigger company."""
    words = _words(question)
    for table in ("by_name", "by_first"):
        found = []
        for i, first in enumerate(words):
            # The question's first word is capitalized anyway: "Total revenue..." is not a name's start.
            if not first[0].isupper() or (table == "by_first" and i == 0):
                continue
            for n in range(1, min(longest, len(words) - i) + 1):
                cik = lookup[table].get(_key(words[i:i + n]))
                if cik is not None:
                    found.append((-n, lookup["rank"][cik], cik))
        if found:
            return min(found)[2]
    for w in words:
        if len(w) >= 2 and w.isupper() and w in lookup["tickers"]:
            return lookup["tickers"][w]
    return None


def find_year(question: str):
    """The latest year named ("2019 compared to 2018" -> 2019: that 10-K holds both), or None."""
    years = [int(y) for y in re.findall(r"\b(?:FY)?((?:19|20)\d\d)\b", question)]
    return max(years) if years else None


def search(client, embedder, question: str, lookup: dict, k: int = 5, collection: str = COLLECTION,
           cik=None, year=None) -> dict:
    """The company and year (shown on the page) and the top-k passages. `cik`/`year`, when given,
    hold whatever the text says: the agent's follow-up searches stay inside the question's company
    and year ("Operating Income" is also a company's name; "2018" isn't in the index). Only when
    they're None are they read from the text."""
    from qdrant_client import models
    cik = cik if cik is not None else find_company(question, lookup)
    year = year if year is not None else find_year(question)
    must = [models.FieldCondition(key=f, match=models.MatchValue(value=v))
            for f, v in (("cik", cik), ("year", year)) if v is not None]
    vector = [float(x) for x in embedder.embed([QUERY_PREFIX + question])[0]]
    hits = client.query_points(collection, query=vector, query_filter=models.Filter(must=must) if must else None,
                               limit=k, with_payload=True).points
    who = lookup["names"].get(cik, {}) if cik is not None else {}
    return {"cik": cik, "company": who.get("company"), "ticker": who.get("ticker"), "year": year,
            "passages": [{**h.payload, "score": h.score} for h in hits]}
