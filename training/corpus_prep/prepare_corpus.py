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
