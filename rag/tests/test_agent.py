import pytest

from rag.agent import parse_step, run

QUESTION = "What was Apple's revenue growth in 2019?"
PASSAGE = {"company": "Apple Inc.", "year": 2019, "section": "7", "chunk": 1,
           "text": "Total net sales decreased 2% or $5.4 billion during 2019 compared to 2018"}


@pytest.mark.parametrize("reply, step", [
    ("Answer: -2%", ("answer", "-2%")),
    ('{"tool": "search", "query": "services net sales"}', ("search", "services net sales")),
    ('{"tool": "calculate", "expression": "5.4 / 265.6 * 100"}', ("calculate", "5.4 / 265.6 * 100")),
    ('Let me search.\n{"tool": "search", "query": "iPhone"}\nResult: made up', ("search", "iPhone")),  # first JSON line
    ("I think it went down.\nAnswer: Net sales fell 2%.", ("answer", "Net sales fell 2%.")),
])
def test_parse_step(reply, step):
    assert parse_step(reply) == step


@pytest.mark.parametrize("reply", ["It went down a bit.", '{"tool": "search"}', '{"tool": "browse", "url": "x"}',
                                   '{"tool": "search", "query": "x"'])
def test_parse_step_errors(reply):
    assert parse_step(reply)[0] == "error"


class Script:
    """A fake model: hands back scripted replies in order, keeps the messages it was sent."""
    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, messages):
        self.sent.append([dict(m) for m in messages])
        return self.replies.pop(0)


BANK = [dict(PASSAGE, chunk=i, text=f"passage {i}: " + "word " * 398) for i in range(20)]   # 400 words each


def fake_search(query, k):
    """Like Qdrant: the same best passages, in the same order, whatever is asked."""
    return BANK[:k]


def test_first_search_is_done_for_the_model():
    model = Script("Answer: -2%")
    run(QUESTION, model, fake_search)
    first = model.sent[0]
    assert first[0]["role"] == "system" and first[1]["role"] == "user"
    assert QUESTION in first[1]["content"] and "passage 4:" in first[1]["content"] and "passage 5:" not in first[1]["content"]
    assert "Apple Inc. 2019 Item 7" in first[1]["content"]          # sources labelled


def test_search_then_calculate_then_answer():
    model = Script('{"tool": "search", "query": "net sales 2018"}',
                   '{"tool": "calculate", "expression": "(260,174 - 265,595) / 265,595 * 100"}',
                   "Answer: -2.04%")
    r = run(QUESTION, model, fake_search)
    assert r["answer"] == "-2.04%" and r["stopped"] == "answer"
    assert [s["kind"] for s in r["steps"]] == ["search", "search", "calculate", "answer"]   # step 0: our search
    assert r["steps"][2]["output"] == "-2.04108"                     # exact arithmetic, 6 significant digits
    assert "passage 5:" in model.sent[1][-1]["content"] and "passage 6:" in model.sent[1][-1]["content"]
    assert model.sent[2][-1]["content"] == "Result: -2.04108"
    assert len(r["passages"]) == 7                                    # every passage the model saw (for the page)


def test_bad_arithmetic_is_an_error_result_not_a_crash():
    model = Script('{"tool": "calculate", "expression": "__import__(\'os\')"}', "Answer: not found")
    r = run(QUESTION, model, fake_search)
    assert r["steps"][1]["output"].startswith("error")
    assert r["answer"] == "not found"


def test_a_garbled_reply_gets_a_correction():
    model = Script("Revenue went down.", "Answer: -2%")
    r = run(QUESTION, model, fake_search)
    assert [s["kind"] for s in r["steps"]] == ["search", "error", "answer"]
    assert "Answer:" in model.sent[1][-1]["content"]                  # the correction says how to reply
    assert r["answer"] == "-2%"


def test_too_many_steps_stops_with_no_answer():
    model = Script(*['{"tool": "search", "query": "more"}'] * 10)
    r = run(QUESTION, model, fake_search, max_steps=5)
    assert r["answer"] is None and r["stopped"] == "max_steps"
    assert len(model.sent) == 5


def test_follow_up_search_skips_passages_already_shown():
    model = Script('{"tool": "search", "query": "x"}', '{"tool": "search", "query": "y"}', "Answer: -2%")
    r = run(QUESTION, model, fake_search)
    assert [p["chunk"] for p in r["passages"]] == [0, 1, 2, 3, 4, 5, 6, 7, 8]   # 5 first, then 2 new each time
    assert "passage 4:" not in model.sent[1][-1]["content"]


def test_word_budget_keeps_the_worst_case_under_the_context_limit():
    """3,600 words of passages; at the worst measured 1.77 tokens per word (number-heavy text), every
    message plus a full 512-token reply still fits vLLM's 8,192."""
    model = Script(*['{"tool": "search", "query": "more"}'] * 10)
    r = run(QUESTION, model, fake_search, max_steps=5)
    assert sum(len(p["text"].split()) for p in r["passages"]) <= 3600
    assert "no room" in model.sent[-1][-1]["content"]
    words = sum(len(m["content"].split()) for m in model.sent[-1])
    assert words * 1.77 + 512 <= 8192


def test_prompt_asks_for_one_signed_number():
    """The vote reads the final number as written, so the prompt asks for just that number, negative
    for a decrease (both models get the same prompt)."""
    from rag.agent import SYSTEM
    assert "Answer: -2.0%" in SYSTEM and "negative" in SYSTEM


def test_answer_splits_into_number_and_explanation():
    """The vote sees only the number; the page shows the sentence (its own numbers can't confuse the vote)."""
    r = run(QUESTION, Script("Answer: -2.0% | Total net sales fell from $265.6 billion to $260.2 billion."), fake_search)
    assert r["answer"] == "-2.0%"
    assert r["explanation"] == "Total net sales fell from $265.6 billion to $260.2 billion."
    r = run(QUESTION, Script("Answer: -2%"), fake_search)
    assert (r["answer"], r["explanation"]) == ("-2%", None)
    r = run(QUESTION, Script(*['{"tool": "search", "query": "x"}'] * 5), fake_search)
    assert (r["answer"], r["explanation"]) == (None, None)


def test_prompt_asks_for_a_short_explanation_after_the_number():
    from rag.agent import SYSTEM
    assert "Answer: -2.0% | " in SYSTEM and "one short sentence" in SYSTEM
