"""Builds a financial pretraining corpus from SEC 10-K filings (EDGAR-CORPUS on
Hugging Face, eloukas/edgar-corpus, Apache-2.0): download, clean, filter,
deduplicate, decontaminate against the fine-tuning eval data, tokenize.

Streams one filing at a time, in two passes (pass 1 takes notes, pass 2
writes the survivors' tokens), so the ~10 GB of raw text never sits in memory.

Run (try a small sample first):
    python3 -m training.corpus_prep.prepare_corpus --years 2019 --max-docs 200
"""

import argparse
import json
import re
import unicodedata

# Whole lines that carry no content: page numbers ("12", "- 12 -", "Page 12", "F-3") and "Table of Contents".
# At most 3 digits, so a year standing alone on a line (a table's column heading, "2019") is kept.
JUNK_LINE = re.compile(r"(?i)^(-?\s*\d{1,3}\s*-?|page\s+\d{1,3}|[a-z]-\d{1,3}|table\s+of\s+contents)$")


def clean(text: str) -> str:
    """NFKC normalization (non-breaking/odd spaces -> plain space, ligatures like "ﬁ" -> "fi"),
    spaces collapsed within each line, junk lines dropped, at most one blank line in a row."""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in unicodedata.normalize("NFKC", text).split("\n")]
    text = "\n".join(ln for ln in lines if not JUNK_LINE.match(ln))
    return re.sub(r"\n{3,}", "\n\n", text).strip()

REPO = "eloukas/edgar-corpus"
# 10-K items in filing order; each section's text already starts with its own heading ("Item 1. Business").
SECTIONS = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]


MIN_WORDS = 200                 # shorter than about a page: an empty shell of headings
MEAN_WORD_LEN = (3, 10)         # outside this: mostly symbols, codes or run-together text
MIN_ALPHA_WORD_FRAC = 0.6       # share of words with a letter in them; numbers-only text is a table dump
# Share of the text (by characters) inside lines that repeat an earlier line. By characters, not
# line count: 10-Ks repeat short page headers ("Notes to Consolidated Financial Statements",
# the company name) and "•" bullets on every page, which made genuine reports fail a per-line rule.
MAX_REPEATED_CHAR_FRAC = 0.2


def quality_problem(text: str):
    """Why a (cleaned) filing is junk, or None if it passes. Gopher-style rules (Rae et al. 2021)."""
    words = text.split()
    if len(words) < MIN_WORDS:
        return "too_few_words"
    if not MEAN_WORD_LEN[0] <= sum(map(len, words)) / len(words) <= MEAN_WORD_LEN[1]:
        return "odd_word_length"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < MIN_ALPHA_WORD_FRAC:
        return "few_alpha_words"
    seen, repeated_chars = set(), 0
    for ln in text.split("\n"):
        if ln in seen:
            repeated_chars += len(ln)
        seen.add(ln)
    if repeated_chars / len(text) > MAX_REPEATED_CHAR_FRAC:
        return "repeated_lines"
    return None


def is_exact_copy(text: str, seen: set) -> bool:
    """True if this exact (cleaned) text was seen before; otherwise remembers it. Stores a 16-byte
    fingerprint (BLAKE2b hash) per filing, not the text: ~60k filings cost a few MB of memory."""
    import hashlib
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    if h in seen:
        return True
    seen.add(h)
    return False


NUM_PERM, SHINGLE = 128, 5  # 128 MinHash values per filing; 5-word overlapping pieces ("shingles")
_rng = __import__("numpy").random.default_rng(0)
_A = _rng.integers(1, 2**63, NUM_PERM, dtype="uint64") | 1  # odd multipliers: 128 independent hash functions
_B = _rng.integers(0, 2**63, NUM_PERM, dtype="uint64")


def minhash(text: str):
    """MinHash signature: for each of 128 hash functions, the smallest hash over the filing's
    5-word shingles. Two filings agree on a given value with probability = their Jaccard similarity
    (shared shingles / all shingles). Vectorized with numpy; uint64 arithmetic wraps around on purpose."""
    import zlib
    import numpy as np
    w = np.array([zlib.crc32(t.encode()) for t in text.lower().split()], dtype="uint64")
    n = max(len(w) - SHINGLE + 1, 1)
    sh = np.zeros(n, dtype="uint64")
    for k in range(min(SHINGLE, len(w))):  # shingle hash = word hashes combined by position
        sh = sh * np.uint64(1_000_003) + w[k:k + n]
    sig = np.full(NUM_PERM, np.iinfo("uint64").max, dtype="uint64")
    for i in range(0, n, 8192):  # blocks bound memory: 8192 shingles x 128 = 8 MB
        sig = np.minimum(sig, (sh[i:i + 8192, None] * _A + _B).min(axis=0))
    return (sig >> np.uint64(32)).astype("uint32")  # top 32 bits: the well-mixed part


BANDS, ROWS = 16, 8      # 16 x 8 = 128: two filings become candidates if any band of 8 values matches
NEAR_COPY = 0.8          # estimated Jaccard similarity at or above which a filing counts as a near-copy


class NearCopyIndex:
    """LSH ("locality-sensitive hashing") over MinHash signatures. Each kept filing is filed under 16
    band keys; a new filing is compared only with filings sharing a band key (likely similar ones),
    not with every filing. Streaming, keep-first: a near-copy of a kept filing is dropped and not
    indexed. Memory: 128 x 4 bytes per kept filing (~20 MB for 40k)."""

    def __init__(self):
        self.buckets = [dict() for _ in range(BANDS)]
        self.sigs = []

    def check_and_add(self, sig):
        """Index of the kept filing this one nearly copies (and its similarity), or None and keep it."""
        keys = [sig[b * ROWS:(b + 1) * ROWS].tobytes() for b in range(BANDS)]
        candidates = {i for b, k in enumerate(keys) for i in self.buckets[b].get(k, ())}
        for i in candidates:
            sim = float((self.sigs[i] == sig).mean())
            if sim >= NEAR_COPY:
                return i, sim
        for b, k in enumerate(keys):
            self.buckets[b].setdefault(k, []).append(len(self.sigs))
        self.sigs.append(sig)
        return None


MIN_SENTENCE_CHARS = 60  # shorter sentences ("in millions", "see note 5") are shared by everyone
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n")  # also splits FinQA's spaced-out " . " sentences


def sentence_keys(text: str) -> set:
    """Long sentences reduced to lowercase letters and digits, so the same sentence compares equal
    however it was spaced or punctuated (FinQA writes "( in millions )", 10-Ks "(in millions)").
    Same idea as qlora_financial/prepare_dataset.py's _long_sentences."""
    keys = (re.sub(r"[^a-z0-9]+", "", s.lower()) for s in SENTENCE_END.split(text))
    return {k for k in keys if len(k) >= MIN_SENTENCE_CHARS}


def load_eval_sentences(paths) -> set:
    """Sentence keys of the fine-tuning eval data (the user message holds the report text)."""
    out = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                out |= sentence_keys(json.loads(line)["messages"][1]["content"])
    return out


MAX_LEAK_COMPANIES = 2  # an eval sentence found in more companies' filings than this is boilerplate


class EvalOverlap:
    """Decontamination in two steps. Pass 1: note() records which eval sentences each filing
    contains, and which companies (CIKs) contain each sentence. Between passes: contaminated() =
    filings sharing at least one *rare* eval sentence (found in <= MAX_LEAK_COMPANIES companies).
    Counting companies, not filings: one company's yearly reports repeat its own sentences, and
    those are all the same leak. Boilerplate (e.g. the Item 5 title, in 971 of 1,359 sample filings)
    is common to many companies and ignored."""

    def __init__(self, eval_sentences: set):
        self.eval, self.by_filing, self.companies = eval_sentences, {}, {}

    def note(self, filing_id: str, cik: str, text: str):
        shared = sentence_keys(text) & self.eval
        if shared:
            self.by_filing[filing_id] = shared
            for s in shared:
                self.companies.setdefault(s, set()).add(cik)

    def contaminated(self) -> set:
        rare = {s for s, c in self.companies.items() if len(c) <= MAX_LEAK_COMPANIES}
        return {f for f, shared in self.by_filing.items() if shared & rare}


def iter_filings(year: int, split: str, max_docs: int = None):
    """Yield one filing at a time from {year}/{split}.jsonl as {"id", "cik", "year", "text"}: the
    non-empty sections joined in order. The file is downloaded once (Hugging Face cache) and read
    line by line, so only one filing is in memory at a time."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, f"{year}/{split}.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                return
            row = json.loads(line)
            parts = [row.get(f"section_{s}") or "" for s in SECTIONS]
            yield {"id": row["filename"], "cik": row["cik"], "year": int(row["year"]),
                   "text": "\n\n".join(p.strip() for p in parts if p.strip())}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2021)))
    ap.add_argument("--out-dir", default="training/corpus_prep/data")
    ap.add_argument("--max-docs", type=int, default=None, help="stop after N filings per year/split (for trying it out)")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"years={args.years} max_docs={args.max_docs} out={args.out_dir}")


if __name__ == "__main__":
    main()
