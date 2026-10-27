"""drift.py: per-question measurements and the drift check, on made-up questions. No GPU."""
import random

from training.qlora_financial.drift import drift, features


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


def test_table_rows_in_every_format_the_sources_use():
    html = "text <table><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></table> more\nQuestion: q?"
    ruler = "text\n\n*****************\n, 2002, 2001\nsales, 5, 4\n*****************\n\nQuestion: q?"
    marks = "text\n• date, x, y\n• 2011, 1, 2\nRow 1: a, 1\nRow 2: b, 2\nQuestion: q?"
    assert [features(p)["table_rows"] for p in (html, ruler, marks)] == [2, 2, 4]  # ConvFinQA, FinQA, FinQA
    assert features("a list:\n- one\n- two\n----------\nQuestion: q?")["table_rows"] == 0  # a lone ruler isn't a table


def made_up(n, seed, doc_words=(50, 150), tables=(0, 5), question_words=(5, 15)):
    """n made-up questions: a document of random words and numbers, a small table, a question."""
    rng = random.Random(seed)
    words = ["revenue", "rose", "the", "cost", "of", "sales", "in", "2019", "$", "12.5"]
    return ["".join(f"row{k}\t{rng.randint(1, 99)}\n" for k in range(rng.randint(*tables)))
            + " ".join(rng.choice(words) for _ in range(rng.randint(*doc_words)))
            + "\nQuestion: " + " ".join(rng.choice(words) for _ in range(rng.randint(*question_words)))
            for _ in range(n)]


def test_drift_same_kind_of_questions_is_quiet():
    summary, _ = drift(made_up(3000, seed=0), made_up(500, seed=1))
    assert summary == {"drift_share": 0.0, "drifted_columns": [], "dataset_drift": False}


def test_drift_names_what_changed_and_alarms_at_half():
    reference = made_up(3000, seed=0)
    longer_q, _ = drift(reference, made_up(500, seed=1, question_words=(30, 40)))
    assert longer_q["drifted_columns"] == ["question_words"] and not longer_q["dataset_drift"]  # 1 of 4: no alarm
    # longer documents and bigger tables; the share of digits moves too (the table's numbers are diluted)
    bigger, _ = drift(reference, made_up(500, seed=1, doc_words=(400, 600), tables=(10, 20)))
    assert bigger["drifted_columns"] == ["document_words", "table_rows", "digit_share"]
    assert bigger["drift_share"] == 0.75 and bigger["dataset_drift"]  # 3 of 4: the alarm
