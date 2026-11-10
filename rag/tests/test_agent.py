import re

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
    assert "Answer: -3.5%" in SYSTEM and "negative" in SYSTEM


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
    assert "Answer: -3.5% | " in SYSTEM and "one short sentence" in SYSTEM


@pytest.mark.parametrize("reply", [
    "Let me think. " + "blah " * 2000 + '\n{"tool": "search", "query": "' + "more " * 300 + '"}',   # rambling, then a long query
    "blah " * 2000,                                                                                   # garbled every round
])
def test_long_replies_dont_pile_up_past_the_context_limit(reply):
    """Review finding 3: replies (up to 512 tokens each) were kept whole, so 4 earlier ones plus a
    full passage budget could pass vLLM's 8,192. Only each reply's essential line is kept; the page
    still gets the full reply."""
    model = Script(*[reply] * 5)
    r = run(QUESTION, model, fake_search, max_steps=5)
    words = sum(len(m["content"].split()) for m in model.sent[-1])
    assert words * 1.77 + 512 <= 8192
    assert all(s["reply"] == reply for s in r["steps"][1:])                 # full reply kept for the page


def test_the_kept_line_is_the_tool_call():
    model = Script('I will look.\n{"tool": "search", "query": "net sales"}\nResult: invented', "Answer: -2%")
    run(QUESTION, model, fake_search)
    assert model.sent[1][2] == {"role": "assistant", "content": '{"tool": "search", "query": "net sales"}'}


@pytest.mark.parametrize("expression, result", [
    ("641 / (1 + 15%)", "557.391"),                       # review finding 4: was 40.0625 (% dropped)
    ("641 * (1 - 15%)", "544.85"),
    ("12.5% * 80", "10"),
    ("(5,421) / 265,595", "0.0204108"),                   # brackets, not an accounting negative
    ("(260,174 - 265,595) / 265,595 * 100", "-2.04108"),  # unchanged
    ("-(5421) / 265595", "-0.0204108"),
])
def test_calculate_reads_percents_and_brackets_as_a_person_would(expression, result):
    from rag.agent import calculate
    assert calculate(expression) == result


def test_prompt_examples_share_nothing_with_the_demo_questions():
    """Found in the first GPU run: the examples were the Apple demo answer and Apple's real net
    sales, so the Apple question was given away and v3 copied 260174 / 265595 into other companies'
    answers. The examples must use a made-up company and numbers from no demo answer or quote."""
    import json
    from pathlib import Path
    from rag.agent import SYSTEM
    demos = json.loads((Path(__file__).parents[1] / "demo_questions.json").read_text())
    numbers = lambda text: {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)} - {10.0} - set(range(1990, 2031))
    leaked = numbers(SYSTEM) & set().union(*(numbers(d["answer"] + " " + d["source"]) for d in demos))
    assert not leaked
    for company in ("Apple", "Netflix", "NVIDIA", "Microsoft"):
        assert company.lower() not in SYSTEM.lower()


def test_prompt_asks_for_percents_as_decimals():
    from rag.agent import SYSTEM
    assert "8% as 0.08" in SYSTEM


@pytest.mark.parametrize("reply, step", [               # review finding 5: real Llama-3 styles
    ("**Answer:** -2% | Net sales fell.", ("answer", "-2% | Net sales fell.")),
    ("answer: -2%", ("answer", "-2%")),
    ('{\n  "tool": "search",\n  "query": "net sales"\n}', ("search", "net sales")),          # spread over lines
    ('I will search: {"tool": "search", "query": "iPhone"}', ("search", "iPhone")),         # text first
    ('{"tool": "search", "query": "a"} {"tool": "search", "query": "b"}', ("search", "a")),  # two: the first
    ('```json\n{"tool": "calculate", "expression": "1 + 1"}\n```', ("calculate", "1 + 1")),
])
def test_parse_step_reads_common_llama_styles(reply, step):
    assert parse_step(reply) == step


def test_a_run_stops_at_the_deadline():
    """Review finding 11: no overall limit, so a stuck vLLM kept the page spinning ~10 minutes."""
    import time
    model = Script(*['{"tool": "search", "query": "x"}'] * 5)
    r = run(QUESTION, model, fake_search, deadline=time.time() - 1)
    assert (r["stopped"], r["answer"], len(model.sent)) == ("time_limit", None, 0)

    def slow(messages):
        time.sleep(0.2)
        return '{"tool": "search", "query": "x"}'
    t = time.time()
    r = run(QUESTION, slow, fake_search, deadline=time.time() + 0.3)
    assert r["stopped"] == "time_limit" and time.time() - t < 0.7
