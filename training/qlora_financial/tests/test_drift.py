"""drift.py: per-question measurements and the drift check, on made-up questions. No GPU."""
from training.qlora_financial.drift import features


def test_features_table_question():
    p = "Table:\nYear\t2019\t2018\nRevenue\t$10\t$8\n\nQuestion: What was the change in revenue?"
    assert features(p) == {"question_words": 6, "document_words": 7, "table_rows": 2,
                           "digit_share": 11 / len(p)}  # 2019, 2018, 10, 8 = 11 digits


def test_features_plain_text_and_conversation_history():
    assert features("Sales rose 5% in 2020.\n\nQuestion: Why did sales rise?")["question_words"] == 4
    f = features("Some report text.\nQuestion: first one?\nAnswer: 3\nQuestion: and the second one?")
    assert (f["document_words"], f["question_words"], f["table_rows"]) == (3, 4, 0)  # doc before the first, question after the last


def test_features_without_a_question_line():
    f = features("just some text 42")
    assert (f["question_words"], f["document_words"]) == (0, 4)
