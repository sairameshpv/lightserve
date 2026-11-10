"""The demo's web server: one question -> the original and the fine-tuned model, side by side.

Both models get the same passages, tools, seeds and settings (best-of-N); only the model differs.
Every page and the /ask endpoint need the password (HTTP basic auth: the browser asks for it),
because during a GPU session the page is on the node's public address.

    RAG_PASSWORD=... python -m rag.app --vllm http://localhost:8000 --device cuda
"""
import json
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from rag import best_of_n, retrieve
from rag.ingest import COLLECTION

MODELS = {"original": "meta-llama/Meta-Llama-3-8B-Instruct", "fine-tuned": "v3"}   # page label -> vLLM model name
PAGE = Path(__file__).parent / "static" / "index.html"
EXPECTED = Path(__file__).parent / "demo_questions.json"   # the demo questions' answers, looked up by hand
MAX_QUESTION = 1000


def _normal(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def load_expected(path) -> dict:
    """The demo questions' checked answers, keyed by the question with case and punctuation ignored.
    No file: no checked answers."""
    path = Path(path)
    return {_normal(e["question"]): e for e in json.loads(path.read_text())} if path.exists() else {}


def verdict(models_out: dict) -> dict:
    """One line comparing the two models' voted answers (same number by the vote's rule = agree)."""
    (a, ma), (b, mb) = models_out.items()
    failed = [(label, m["error"]) for label, m in models_out.items() if m.get("failed")]
    if len(failed) == 2:   # a server problem, not the models' answers (review finding 2)
        return {"agree": False, "text": f"Both models failed, so there's nothing to compare. Error: {failed[0][1]}"}
    if failed:
        return {"agree": False, "text": f"The {failed[0][0]} model failed, so there's nothing to compare. "
                                        f"Error: {failed[0][1]}"}
    va, vb = ma["vote"], mb["vote"]
    said = lambda v: f"{v['answer']} ({v['votes']} of {v['n']} runs)"
    if va["answer"] is None and vb["answer"] is None:
        return {"agree": False, "text": "Neither model found an answer."}
    if va["answer"] is None or vb["answer"] is None:
        who, v = (b, vb) if va["answer"] is None else (a, va)
        return {"agree": False, "text": f"Only the {who} model found an answer: {said(v)}."}
    if best_of_n.same_answer(va["answer"], vb["answer"]):
        return {"agree": True, "text": f"Both models give the same answer: {va['answer']}."}
    return {"agree": False, "text": f"The models disagree: {a} says {said(va)}, "
                                    f"{b} says {vb['answer']} ({vb['votes']} of {vb['n']})."}


class Ask(BaseModel):
    question: str


BANNER = '<div class="banner">Rehearsal: scripted fake models, not real answers</div>'


def create_app(client, embedder, lookup: dict, chat, password: str, models: dict = MODELS, n: int = 5,
               collection: str = COLLECTION, rehearsal: bool = False, expected: dict = None) -> FastAPI:
    """`chat(model, messages, temperature, top_p, seed) -> reply text`. `rehearsal`: every page
    says the models are scripted fakes."""
    if not password:
        raise ValueError("set a password (RAG_PASSWORD): the page is on a public address")
    # No /docs, /redoc or /openapi.json: they would open without the password.
    app = FastAPI(title="RAG demo", docs_url=None, redoc_url=None, openapi_url=None)
    basic = HTTPBasic(auto_error=False)
    app.state.expected = expected or {}

    def check(creds: HTTPBasicCredentials = Depends(basic)):
        if creds is None or not secrets.compare_digest(creds.password.encode(), password.encode()):
            raise HTTPException(401, "password needed", headers={"WWW-Authenticate": "Basic"})

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(check)])
    def page():
        return PAGE.read_text().replace("<!--BANNER-->", BANNER if rehearsal else "")

    @app.post("/ask", dependencies=[Depends(check)])
    def ask(body: Ask):
        q = body.question.strip()
        if not q or len(q) > MAX_QUESTION:
            raise HTTPException(400, f"ask a question of 1-{MAX_QUESTION} characters")
        start = time.time()
        cik, year = retrieve.find_company(q, lookup), retrieve.find_year(q)
        who = lookup["names"].get(cik, {}) if cik is not None else {}
        cache, lock = {}, threading.Lock()

        def search(query, k):
            """The same query -> the same passages for all 2 x N runs (one embedding, one lookup)."""
            with lock:
                if (query, k) not in cache:
                    cache[query, k] = retrieve.search(client, embedder, query, lookup, k=k, collection=collection,
                                                      cik=cik, year=year)["passages"]
                return [dict(p) for p in cache[query, k]]

        out = {"question": q, "cik": cik, "company": who.get("company"), "ticker": who.get("ticker"),
               "year": year, "notes": [], "models": {}, "expected": app.state.expected.get(_normal(q))}
        if cik is None:
            out["notes"].append("No company recognised in the question: searching every company's reports.")
        if year is None:
            out["notes"].append("No year named: searching every year in the index (fiscal 2019).")
        if cik is not None and not search(q, 5):
            out["notes"].append(f"{who.get('company')}'s fiscal {year or ''} 10-K text isn't in the dataset "
                                "(its Items 7 and 8 are empty or missing), so neither model was asked.")
            out["seconds"] = round(time.time() - start, 2)
            return out

        def one(label):
            t = time.time()
            r = best_of_n.best_of_n(q, lambda m, **s: chat(models[label], m, **s), search, n=n)
            errs = [run["error"] for run in r["runs"] if run["stopped"] == "error"]
            return label, {"model": models[label], **r, "seconds": round(time.time() - t, 2),
                           "errors": len(errs), "failed": len(errs) == n, "error": errs[0][:300] if errs else None}

        with ThreadPoolExecutor(max_workers=len(models)) as pool:   # both models at once
            out["models"] = dict(pool.map(one, models))
        out["verdict"] = verdict(out["models"])
        for m in out["models"].values():         # right or wrong against the checked answer, if any
            m["correct"] = (best_of_n.same_answer(m["vote"]["answer"], out["expected"]["answer"])
                            if out["expected"] and not m["failed"] else None)
        out["seconds"] = round(time.time() - start, 2)
        return out

    return app


def main():
    import argparse
    import os

    import uvicorn
    from qdrant_client import QdrantClient

    from rag import llm
    from rag.ingest import Embedder
    from rag.workload import WorkloadLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm", default="http://localhost:8000")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--names", default="rag/data/company_tickers.json")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--device", default="cpu", help="cuda on the node")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve on the node's public address")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n", type=int, default=5, help="runs per model for the vote")
    ap.add_argument("--rehearsal", action="store_true", help="--vllm is a scripted fake: say so on the page")
    ap.add_argument("--workload", help="log every model call here (JSONL), for replay with the A/B tool")
    a = ap.parse_args()

    log = WorkloadLog(a.workload) if a.workload else None

    def chat(model, messages, temperature, top_p, seed):
        reply = llm.chat(messages, model=model, base_url=a.vllm, temperature=temperature, top_p=top_p,
                         seed=seed, max_tokens=512)
        if log:
            log.record(engine="vllm", model=model, messages=messages, reply=reply, temperature=temperature,
                       top_p=top_p, seed=seed, max_tokens=512)
        return reply["text"]

    app = create_app(QdrantClient(url=a.qdrant, timeout=60), Embedder(a.device), retrieve.load_lookup(a.names),
                     chat, password=os.environ.get("RAG_PASSWORD", ""), n=a.n, collection=a.collection,
                     rehearsal=a.rehearsal, expected=load_expected(EXPECTED))
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
