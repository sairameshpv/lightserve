"""A question -> a few rounds of search / calculate -> one answer.

The tools are described in plain words in the prompt, not through vLLM's built-in tool format:
neither model (Llama-3-8B-Instruct, or its fine-tune v3) was trained for that, so a plain format
treats both the same. Each reply is one JSON line (a tool) or "Answer: ...".
"""
import json
import re
import time

from training.qlora_financial.calculator import eval_formula

SYSTEM = """You answer questions about companies' annual reports (10-K filings), using only the passages you are given.
You have two tools. To use one, reply with exactly one line of JSON and nothing else:
{"tool": "search", "query": "<what to look for in the report>"}
{"tool": "calculate", "expression": "<plain arithmetic, e.g. (46500 - 48200) / 48200 * 100>"}
Write a percent inside an expression as a decimal: 8% as 0.08.
The result comes back in the next message. When you know the answer, reply with one line: the number
and its unit (negative for a decrease), then "|" and one short sentence saying what the number is.
For example:
Answer: -3.5% | Total net sales fell from $48.2 billion in 2019 to $46.5 billion in 2020.
Answer: $46.5 billion | Acme Corp's total net sales in fiscal 2020.
If the passages don't hold the answer, reply: Answer: not found | <what is missing>"""
CORRECTION = 'Reply with exactly one line: a JSON tool call such as {"tool": "search", "query": "..."}, or "Answer: ...".'
TOOLS = {"search": "query", "calculate": "expression"}   # tool -> the field it needs


_ANSWER = re.compile(r"^[\s*_#>`]*answer[\s*_]*:[\s*_]*(.*)$", re.I | re.M)   # "**Answer:** x", "answer: x"


def parse_step(reply: str) -> tuple:
    """A reply -> ("answer"|"search"|"calculate", text), or ("error", why). Whichever comes first:
    an "Answer:" line (any case, markdown stars ignored) or a JSON object (wherever it starts, even
    spread over lines or after some text; the first of several). Anything after it, e.g. a made-up
    "Result:", is ignored."""
    answer, brace = _ANSWER.search(reply), reply.find("{")
    if answer and (brace < 0 or answer.start() < brace):
        return "answer", answer[1].strip()
    if brace < 0:
        return "error", "no tool call and no answer"
    try:
        call, _ = json.JSONDecoder().raw_decode(reply[brace:])
    except json.JSONDecodeError:
        return ("answer", answer[1].strip()) if answer else ("error", "not valid JSON")
    tool = call.get("tool") if isinstance(call, dict) else None
    if tool not in TOOLS or not isinstance(call.get(TOOLS[tool]), str):
        return "error", f"unknown tool or missing field: {json.dumps(call)[:80]}"
    return tool, call[TOOLS[tool]]


def format_passages(passages: list) -> str:
    return "\n\n".join(f"[{i}] {p['company']} {p['year']} Item {p['section']}: {p['text']}"
                       for i, p in enumerate(passages, 1)) or "(no passages found)"


def calculate(expression: str) -> str:
    """Exact arithmetic, read as a person writes it: "15%" is 0.15 (the finance calculator drops "%",
    TAT-QA's convention, so "641 / (1 + 15%)" gave 40.06), and "(5,421)" is just a bracket, not an
    accounting negative."""
    expression = re.sub(r"\((\s*\d[\d,]*(?:\.\d+)?\s*)\)", r"\1", expression)
    expression = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*%", r"(\1/100)", expression)
    value = eval_formula(expression)
    return f"{value:.6g}" if value is not None else "error: not plain arithmetic (use numbers, + - * / and brackets)"


def _kept(kind: str, value: str, reply: str) -> str:
    """What stays in the conversation of a reply: its tool call as one clean JSON line (the field
    capped at 250 characters), or the first 200 characters of a garbled one. A reply can be up to
    512 tokens; kept whole, 4 of them plus the passage budget could pass the 8,192-token limit."""
    if kind in TOOLS:
        return json.dumps({"tool": kind, TOOLS[kind]: value[:250]})
    return reply.strip()[:200]


def _key(p: dict) -> tuple:
    return p.get("cik"), p.get("year"), p.get("section"), p.get("chunk")


def run(question: str, chat_fn, search_fn, max_steps: int = 5, first_k: int = 5, more_k: int = 2,
        word_budget: int = 3600, deadline: float = None) -> dict:
    """`chat_fn(messages) -> reply text` (one model, its sampling settings fixed by the caller);
    `search_fn(query, k) -> the top-k passages`. The question's own search (first_k passages) is
    done first, for both models alike; each follow-up adds up to more_k passages not shown before.
    At most `word_budget` words of passages per run: at the worst measured 1.77 Llama-3 tokens per
    word, everything still fits vLLM's 8,192-token limit with room for the reply.
    `deadline` (a time.time()): no new round after it; the run stops with "time_limit".
    Returns {"answer" (the number part, None if none), "explanation" (after "|"), "stopped" ("answer"|"max_steps"), "steps", "passages"}."""
    passages = search_fn(question, first_k)
    seen, used = {_key(p) for p in passages}, sum(len(p["text"].split()) for p in passages)
    steps = [{"kind": "search", "input": question, "output": format_passages(passages)}]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{steps[0]['output']}"}]
    for _ in range(max_steps):
        if deadline is not None and time.time() >= deadline:
            return {"answer": None, "explanation": None, "stopped": "time_limit", "steps": steps, "passages": passages}
        reply = chat_fn(messages)
        kind, value = parse_step(reply)
        messages.append({"role": "assistant", "content": _kept(kind, value, reply)})
        if kind == "answer":
            number, _, why = value.partition("|")
            steps.append({"kind": kind, "input": value, "output": value, "reply": reply})
            return {"answer": number.strip(), "explanation": why.strip() or None, "stopped": "answer",
                    "steps": steps, "passages": passages}
        if kind == "search":
            new = [p for p in search_fn(value, more_k + len(seen)) if _key(p) not in seen][:more_k]
            fits = []
            for p in new:
                if used + len(p["text"].split()) <= word_budget:
                    fits.append(p)
                    used += len(p["text"].split())
            seen |= {_key(p) for p in fits}
            passages = passages + fits
            out = (format_passages(fits) if fits else
                   "(no room for more passages: answer from the passages above)" if new else
                   "(no new passages: answer from the passages above)")
        elif kind == "calculate":
            out = calculate(value)
        else:
            out = f"({value}) {CORRECTION}"
        steps.append({"kind": kind, "input": value, "output": out, "reply": reply})
        messages.append({"role": "user", "content": out if kind == "error" else f"Result: {out}"})
    return {"answer": None, "explanation": None, "stopped": "max_steps", "steps": steps, "passages": passages}
