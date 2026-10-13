"""calculator.py on hand-made steps. Pure Python, no GPU."""
from training.qlora_financial.calculator import apply_calculator, eval_formula, run_finqa_steps


def close(a, b):
    return abs(a - b) < 1e-9


def test_finqa_steps():
    assert close(run_finqa_steps(["divide 151 and 211."]), 151 / 211)  # v2 wrote 0.7168
    assert close(run_finqa_steps(["subtract 5363 and 7983.", "divide the result of step 1 and 7983."]),
                 (5363 - 7983) / 7983)  # chaining
    assert close(run_finqa_steps(["multiply 450 and 6.625%."]), 450 * 0.06625)  # FinQA percent rule
    assert close(run_finqa_steps(["add $1,341.2 and (14,299)."]), 1341.2 - 14299)  # $, commas, accounting negative
    assert close(run_finqa_steps(["raise to the power of 1.05 and 2."]), 1.05 ** 2)


def test_finqa_steps_not_calculable():
    assert run_finqa_steps(["take the max of gross margin and none."]) is None  # not arithmetic
    assert run_finqa_steps(["average equity securities and none."]) is None
    assert run_finqa_steps(["divide 5 and 0."]) is None
    assert run_finqa_steps(["divide the result of step 3 and 2."]) is None  # no such step
    assert run_finqa_steps(["add 1 and 2.", "divide the result of step 0 and 2."]) is None
    assert run_finqa_steps([]) is None


def test_eval_formula():
    assert close(eval_formula("(152,231 - 109,283)/109,283"), (152231 - 109283) / 109283)
    assert close(eval_formula("11-(-46)"), 57)
    assert close(eval_formula("(14,299) + 100"), -14199)  # accounting negative
    assert close(eval_formula("(596.2 +325.4)/1,341.2"), (596.2 + 325.4) / 1341.2)  # bracketed sum is not negative
    assert close(eval_formula("$5.2% - 3.1%"), 2.1)  # stays in the table's percent units


def test_eval_formula_refuses_non_arithmetic():
    assert eval_formula("__import__('os').system('ls')") is None  # never runs code
    assert eval_formula("2 ** 10") is None  # power isn't in TAT-QA formulas
    assert eval_formula("revenue - cost") is None
    assert eval_formula("1/0") is None
    assert eval_formula("") is None


def test_apply_calculator():
    finqa = "Step 1: divide 151 and 211.\nAnswer: 0.7168"
    assert apply_calculator(finqa) == "Step 1: divide 151 and 211.\nAnswer: 0.71564"
    tat = "Step 1: compute 20,642-15,221.\nAnswer: 5421 thousand"
    assert apply_calculator(tat).endswith("Answer: 5421.00000 thousand")  # keeps the unit word
    ratio = "Step 1: compute (152,231 - 109,283)/109,283.\nAnswer: 39.34 percent"
    assert apply_calculator(ratio).endswith("Answer: 39.29980 percent")  # ratio -> percent
    points = "Step 1: compute 5.2% - 3.1%.\nAnswer: 2.1 percent"
    assert apply_calculator(points).endswith("Answer: 2.10000 percent")  # already in percent units


def test_apply_calculator_leaves_alone():
    for text in ("Answer: 0.70103",  # no steps (ConvFinQA style)
                 "Step 1: take the max of revenue and none.\nAnswer: 5",  # not arithmetic
                 "Step 1: count the matching items: 2019; 2018.\nAnswer: 2",
                 "Step 1: divide 151 and 211.\nStep 2: add the result of step 1 and"):  # cut off, no Answer
        assert apply_calculator(text) == text
