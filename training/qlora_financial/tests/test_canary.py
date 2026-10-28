"""canary.py: the fixed canary questions and how their answers are graded. No GPU, no server."""
from training.qlora_financial.canary import grade, pick


def questions(n):
    return [{"id": i, "messages": [{}, {"content": f"q{i}"}], "gold": f"Answer: {i}"} for i in range(n)]


def test_pick_is_the_same_50_every_time():
    qs = questions(300)
    first = pick(qs, n=50, seed=0)
    assert len(first) == 50 and len({q["id"] for q in first}) == 50
    assert first == pick(qs, n=50, seed=0) and first != pick(qs, n=50, seed=1)


def test_grade_uses_the_calculator_and_the_fair_checker():
    qs = [{"gold": "Step 1: divide 8475 and 22110.\nAnswer: 0.38331"}, {"gold": "Answer: 0.06625"},
          {"gold": "2018"}, {"gold": "Answer: 5"}]
    answers = ["Step 1: divide 8475 and 22110.\nAnswer: 0.4",  # wrong number, right step: the calculator fixes it
               "Answer: 0.0665",  # 0.4% off: right under the old 1% checker, wrong under the fair one
               "2018",
               "I don't know"]
    assert grade(qs, answers) == 0.5
