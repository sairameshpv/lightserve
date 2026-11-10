"""10-K text -> tagged, overlapping chunks -> Qdrant (HNSW).

The documents are EDGAR-CORPUS (Hugging Face `eloukas/edgar-corpus`): one 10-K per record, its
text split by section (`section_7` = Item 7, management's discussion; `section_8` = Item 8, the
financial statements), with the company's SEC ID (`cik`) and the `year`. Company names come from
the SEC's ticker list (`company_tickers.json`: SEC ID -> name, ticker).
"""
import json
from pathlib import Path

SECTIONS = ("7", "8")      # management's discussion + the financial statements: where the numbers are


def chunk_words(text: str, size: int = 400, overlap: int = 50) -> list:
    """Chunks of `size` words, each starting `overlap` words before the previous one ended (so a
    sentence cut at a boundary is whole in one of the two)."""
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_names(path) -> dict:
    """SEC ID -> {"company", "ticker"}; a company listed under several tickers keeps the first."""
    names = {}
    for row in json.loads(Path(path).read_text()).values():
        names.setdefault(int(row["cik_str"]), {"company": row["title"], "ticker": row["ticker"]})
    return names


def filing_chunks(filing: dict, names: dict, size: int = 400, overlap: int = 50) -> list:
    """One 10-K -> its Items 7 and 8 as chunks, each tagged with company, ticker, year, section."""
    cik = int(filing["cik"])
    who = names.get(cik, {"company": f"CIK {cik}", "ticker": None})
    out = []
    for section in SECTIONS:
        for i, text in enumerate(chunk_words(filing.get(f"section_{section}") or "", size, overlap)):
            out.append({"cik": cik, **who, "year": int(filing["year"]), "section": section,
                        "chunk": i, "text": text})
    return out


# --- embedding + Qdrant ------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 numbers per chunk
COLLECTION = "tenk"
SOURCE = "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{year}/{split}.jsonl"
SPLITS = ("train", "validate", "test")  # a company's filing can be in any of the three


class Embedder:
    """sentence-transformers on the GPU (the node) or the CPU (the Mac's tests). The SAME model and
    settings must embed the chunks and, later, the questions, or search breaks. (fastembed on the
    Mac's CPU managed 3.8 chunks/s: ~17 h for the year, so the build runs on the node's GPU.)"""

    def __init__(self, device: str = "cpu", batch_size: int = 128):
        from sentence_transformers import SentenceTransformer
        self.model, self.batch_size = SentenceTransformer(EMBED_MODEL, device=device), batch_size

    def embed(self, texts: list) -> list:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True)


def make_collection(client, name: str = COLLECTION, dim: int = 384):
    """Cosine distance, HNSW set explicitly (m=16 links per node, ef_construct=100 while building),
    and payload indexes so filtering by company and year is fast."""
    from qdrant_client import models
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                             hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100))
    client.create_payload_index(name, "cik", models.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "year", models.PayloadSchemaType.INTEGER)


def add_chunks(client, embedder, chunks: list, start_id: int, name: str = COLLECTION, batch: int = 256) -> int:
    """Embed and store chunks (each chunk's payload = its tags + text). Returns the next free id."""
    from qdrant_client import models
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vectors = list(embedder.embed([c["text"] for c in part]))
        client.upsert(name, points=[models.PointStruct(id=start_id + j, vector=[float(x) for x in v], payload=c)
                                    for j, (c, v) in enumerate(zip(part, vectors))])
        start_id += len(part)
    return start_id


def iter_filings(stream):
    """One filing per line of a JSONL byte stream (a download being read, or a file)."""
    for line in stream:
        if line.strip():
            yield json.loads(line)


def build(streams, names: dict, client, embedder, name: str = COLLECTION, batch: int = 512,
          limit: int = None, report_every: int = 250) -> dict:
    """Stream filings -> chunks -> embeddings -> Qdrant, without keeping the filings on disk.
    `limit` stops after that many filings (the sample run)."""
    import time
    filings, next_id, pending, t0 = 0, 0, [], time.time()
    for stream in streams:
        for filing in iter_filings(stream):
            pending += filing_chunks(filing, names)
            filings += 1
            if len(pending) >= batch:
                next_id = add_chunks(client, embedder, pending, next_id, name, batch)
                pending = []
            if filings % report_every == 0:
                print(f"{filings} filings, {next_id} chunks, {time.time() - t0:.0f} s", flush=True)
            if limit and filings >= limit:
                break
        if limit and filings >= limit:
            break
    if pending:
        next_id = add_chunks(client, embedder, pending, next_id, name, batch)
    return {"filings": filings, "chunks": next_id}


def main():
    import argparse
    import urllib.request
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--names", default="rag/data/company_tickers.json", help="the SEC's ticker list")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--device", default="cpu", help="cuda on the node")
    ap.add_argument("--limit", type=int, help="stop after N filings (the sample run)")
    a = ap.parse_args()
    from qdrant_client import QdrantClient
    client, embedder = QdrantClient(url=a.qdrant, timeout=120), Embedder(a.device)
    make_collection(client, a.collection)
    streams = (urllib.request.urlopen(urllib.request.Request(SOURCE.format(year=a.year, split=s),
               headers={"User-Agent": "lightserve-rag-demo"}), timeout=120) for s in SPLITS)
    print(build(streams, load_names(a.names), client, embedder, a.collection, limit=a.limit), flush=True)


if __name__ == "__main__":
    main()
