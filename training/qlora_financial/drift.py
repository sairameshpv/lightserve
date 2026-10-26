"""Query drift: do the questions being asked still look like the ones the model was
trained on? Each question becomes a few measurements taken from its text alone (real
users don't say which dataset their question resembles), and Evidently compares this
window's measurements with the training questions', column by column."""


def features(prompt: str) -> dict:
    """The measurements of one question. Prompts are a document (text and/or a table),
    then `Question: ...`; ConvFinQA puts earlier turns in between, so the document ends
    at the first `Question:` and the question starts after the last one."""
    document = prompt.split("Question:")[0]
    question = prompt.rsplit("Question:", 1)[1] if "Question:" in prompt else ""
    return {"question_words": len(question.split()), "document_words": len(document.split()),
            "table_rows": sum("\t" in line or " | " in line for line in prompt.splitlines()),  # TAT-QA tabs, FinQA bars
            "digit_share": sum(c.isdigit() for c in prompt) / max(len(prompt), 1)}
