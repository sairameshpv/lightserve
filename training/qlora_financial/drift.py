"""Query drift: do the questions being asked still look like the ones the model was
trained on? Each question becomes a few measurements taken from its text alone (real
users don't say which dataset their question resembles), and Evidently compares this
window's measurements with the training questions', column by column."""
import re

RULER = re.compile(r"^[*=-]{10,}$")  # FinQA puts some tables between two of these
ROW_MARK = re.compile(r"^(• |Row \d+: )")  # FinQA's other row styles


def features(prompt: str) -> dict:
    """The measurements of one question. Prompts are a document (text and/or a table),
    then `Question: ...`; ConvFinQA puts earlier turns in between, so the document ends
    at the first `Question:` and the question starts after the last one."""
    document = prompt.split("Question:")[0]
    question = prompt.rsplit("Question:", 1)[1] if "Question:" in prompt else ""
    return {"question_words": len(question.split()), "document_words": len(document.split()),
            "table_rows": table_rows(prompt),
            "digit_share": sum(c.isdigit() for c in prompt) / max(len(prompt), 1)}


def table_rows(prompt: str) -> int:
    """Table rows in any of the sources' formats: ConvFinQA's web-table code (<tr>), TAT-QA's
    tabs, FinQA's ` | ` bars, `• `/`Row N: ` lines, or lines between two ruler lines. FinQA
    rows written as plain comma lines with no marker (about 4% of FinQA) aren't counted."""
    n, block = prompt.count("<tr>"), None
    for line in prompt.splitlines():
        if RULER.match(line.strip()):
            n, block = (n + block, None) if block is not None else (n, 0)  # closing ruler: count the block
        elif block is not None:
            block += 1
        elif "\t" in line or " | " in line or ROW_MARK.match(line):
            n += 1
    return n
