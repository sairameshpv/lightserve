import json
import threading
import zlib

import pytest
from fastapi.testclient import TestClient

from rag.app import MODELS, create_app
from rag.ingest import add_chunks, make_collection
from rag.retrieve import load_lookup

APPLE, AMAZON, INTEL = 320193, 1018724, 50863
AUTH = ("demo", "s3cret")


class FakeEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 384
            for w in t.lower().split():
                v[zlib.crc32(w.encode()) % 384] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


class FakeModels:
    """Plays both models: v3 always says -2.0%; the original says -3% for seeds 0-1, -2% after."""
    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def __call__(self, model, messages, temperature, top_p, seed):
        with self.lock:
            self.calls.append({"model": model, "seed": seed, "temperature": temperature, "top_p": top_p,
                               "first": messages[1]["content"]})
        if model == MODELS["fine-tuned"]:
            return "Answer: -2.0%"
        return "Answer: -3%" if seed < 2 else "Answer: -2%"


@pytest.fixture
def app(tmp_path):
    from qdrant_client import QdrantClient
    rows = [(APPLE, "AAPL", "Apple Inc."), (AMAZON, "AMZN", "AMAZON COM INC"), (INTEL, "INTC", "INTEL CORP")]
    path = tmp_path / "tickers.json"
    path.write_text(json.dumps({str(i): {"cik_str": c, "ticker": t, "title": n} for i, (c, t, n) in enumerate(rows)}))
    client = QdrantClient(":memory:")
    make_collection(client, "t")
    chunks = [{"cik": c, "company": n, "ticker": t, "year": 2019, "section": "7", "chunk": i,
               "text": f"{n} total net sales passage {i}"}
              for c, t, n in ((APPLE, "AAPL", "Apple Inc."), (AMAZON, "AMZN", "AMAZON COM INC")) for i in range(8)]
    add_chunks(client, FakeEmbedder(), chunks, 0, "t")
    models = FakeModels()
    web = TestClient(create_app(client, FakeEmbedder(), load_lookup(path), models, password=AUTH[1], collection="t"))
    return web, models


def test_both_models_answer_with_their_own_vote(app):
    web, models = app
    r = web.post("/ask", json={"question": "What was Apple's revenue growth in 2019?"}, auth=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert (body["company"], body["ticker"], body["year"]) == ("Apple Inc.", "AAPL", 2019)
    assert list(body["models"]) == ["original", "fine-tuned"]
    ft, orig = body["models"]["fine-tuned"], body["models"]["original"]
    assert ft["model"] == "v3" and (ft["vote"]["answer"], ft["vote"]["votes"], ft["vote"]["n"]) == ("-2.0%", 5, 5)
    assert (orig["vote"]["answer"], orig["vote"]["votes"]) == ("-2%", 3)
    assert len(orig["runs"]) == 5 and orig["runs"][0]["steps"][0]["kind"] == "search"
    assert all(p["company"] == "Apple Inc." for p in orig["runs"][0]["passages"])


def test_same_passages_seeds_and_settings_for_both_models(app):
    web, models = app
    web.post("/ask", json={"question": "What was Apple's revenue growth in 2019?"}, auth=AUTH)
    for name in MODELS.values():
        mine = [c for c in models.calls if c["model"] == name]
        assert sorted(c["seed"] for c in mine) == [0, 1, 2, 3, 4]
        assert {(c["temperature"], c["top_p"]) for c in mine} == {(0.7, 0.95)}
    assert len({c["first"] for c in models.calls}) == 1           # one identical first message for all 10 runs


def test_a_company_with_no_text_gets_a_note_and_no_model_calls(app):
    web, models = app
    body = web.post("/ask", json={"question": "What was Intel's revenue in 2019?"}, auth=AUTH).json()
    assert body["company"] == "INTEL CORP" and body["models"] == {}
    assert any("isn't in the dataset" in n for n in body["notes"])
    assert models.calls == []


def test_no_company_named_is_said_on_the_page(app):
    body = app[0].post("/ask", json={"question": "What was total revenue growth in 2019?"}, auth=AUTH).json()
    assert any("every company" in n for n in body["notes"]) and len(body["models"]) == 2


def test_password_required(app):
    web, _ = app
    assert web.get("/").status_code == 401
    assert web.get("/", auth=("demo", "wrong")).status_code == 401
    assert web.post("/ask", json={"question": "q"}).status_code == 401
    page = web.get("/", auth=AUTH)
    assert page.status_code == 200 and "<textarea" in page.text and "/ask" in page.text
    assert web.get("/health").json() == {"ok": True}                # no password: for start-up checks


def test_empty_or_huge_question_is_refused(app):
    web, _ = app
    assert web.post("/ask", json={"question": "  "}, auth=AUTH).status_code == 400
    assert web.post("/ask", json={"question": "x" * 1001}, auth=AUTH).status_code == 400


def test_no_password_no_server():
    with pytest.raises(ValueError):
        create_app(None, None, {}, None, password="")


def test_rehearsal_banner_only_in_a_rehearsal(app, tmp_path):
    """Scripted fake models must never be mistaken for real answers."""
    assert "Rehearsal" not in app[0].get("/", auth=AUTH).text
    web = TestClient(create_app(None, FakeEmbedder(), {}, FakeModels(), password=AUTH[1], rehearsal=True))
    assert "Rehearsal: scripted fake models, not real answers" in web.get("/", auth=AUTH).text


def test_verdict_when_the_models_agree(app):
    body = app[0].post("/ask", json={"question": "What was Apple's revenue growth in 2019?"}, auth=AUTH).json()
    assert body["verdict"] == {"agree": True, "text": "Both models give the same answer: -2%."}


@pytest.mark.parametrize("v3_says, agree, text", [
    ("Answer: -5%", False, "The models disagree: original says -2% (3 of 5 runs), fine-tuned says -5% (5 of 5)."),
    ("Answer: not found", False, "Only the original model found an answer: -2% (3 of 5 runs)."),
])
def test_verdict_when_they_differ(app, monkeypatch, v3_says, agree, text):
    web, models = app
    real = FakeModels.__call__
    monkeypatch.setattr(FakeModels, "__call__", lambda self, model, *a, **k:
                        v3_says if model == MODELS["fine-tuned"] else real(self, model, *a, **k))
    body = web.post("/ask", json={"question": "What was Apple's revenue growth in 2019?"}, auth=AUTH).json()
    assert body["verdict"] == {"agree": agree, "text": text}


EXPECTED = [{"question": "What was Apple's revenue growth in 2019?", "answer": "-2%",
             "source": "Apple Inc. fiscal 2019 10-K, Item 7: \"Total net sales decreased 2% or $5.4 billion\""}]


@pytest.fixture
def checked(app, tmp_path):
    from rag.app import load_expected
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(EXPECTED))
    web, models = app
    web.app.state.expected = load_expected(path)
    return web, models


def test_a_demo_question_shows_the_correct_answer_and_marks_each_model(checked, monkeypatch):
    web, _ = checked
    real = FakeModels.__call__
    monkeypatch.setattr(FakeModels, "__call__", lambda self, model, *a, **k:
                        "Answer: -5%" if model == MODELS["fine-tuned"] else real(self, model, *a, **k))
    body = web.post("/ask", json={"question": "what was apple's revenue growth in 2019"}, auth=AUTH).json()
    assert body["expected"] == EXPECTED[0]                         # case and punctuation ignored
    assert body["models"]["original"]["correct"] is True           # -2% vs -2%
    assert body["models"]["fine-tuned"]["correct"] is False        # -5% vs -2%


def test_other_questions_have_no_checked_answer(checked):
    body = checked[0].post("/ask", json={"question": "What was Apple's net income in 2019?"}, auth=AUTH).json()
    assert body["expected"] is None and body["models"]["original"]["correct"] is None


def test_missing_demo_file_means_no_checked_answers(tmp_path):
    from rag.app import load_expected
    assert load_expected(tmp_path / "nope.json") == {}
