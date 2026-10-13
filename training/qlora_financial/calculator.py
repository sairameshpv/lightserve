"""Exact arithmetic for v2's written steps. v2 picks the right method but
slips doing division in its head (0.7168 for 151/211 = 0.71564); its steps
use prepare_dataset.py's fixed formats, so a program can redo the maths:
    FinQA:  "Step 1: subtract 5363 and 7983." / "divide the result of step 1 and 7983."
    TAT-QA: "Step 1: compute (152,231 - 109,283)/109,283."
"""

import ast
import operator
import re

_FINQA_OPS = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul,
              "divide": operator.truediv, "raise to the power of": operator.pow}
_FINQA_STEP_RE = re.compile(r"(add|subtract|multiply|divide|raise to the power of) (.+?) and (.+?)\.?$")


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _walk(node) -> float:
    """Numbers, + - * /, unary minus/plus, brackets -- anything else raises
    (no names, calls or attributes: this never runs code from the text)."""
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_walk(node.left), _walk(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return -_walk(node.operand) if isinstance(node.op, ast.USub) else _walk(node.operand)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    raise ValueError(f"not plain arithmetic: {ast.dump(node)[:60]}")


def eval_formula(expr: str):
    """A TAT-QA 'compute' formula, or None if it isn't plain arithmetic.
    Commas, $ and % are dropped (TAT-QA formulas work in the table's own
    units); a lone bracketed number is an accounting negative: (14,299)."""
    e = expr.replace(",", "").replace("$", "").replace("%", "")
    e = re.sub(r"\((\d+(?:\.\d+)?)\)", r"(-\1)", e)
    try:
        return _walk(ast.parse(e, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _number(text: str) -> float:
    """'1,341.2' / '$5' / '(14,299)' (accounting negative) / '6.625%' (-> 0.06625, FinQA's rule)."""
    s = text.strip().replace(",", "").replace("$", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    value = float(s[:-1]) / 100 if s.endswith("%") else float(s)
    return -value if negative else value


def calculate(answer: str):
    """(value, TAT-QA formula or None) from the answer's own steps, or None
    if there are no calculable steps or no 'Answer:' line (e.g. cut off at
    max_tokens)."""
    steps = [s.strip() for s in re.findall(r"Step \d+: (.*)", answer)]
    if not steps or "Answer:" not in answer:
        return None
    formula = steps[0][len("compute "):].rstrip(".") if steps[0].startswith("compute ") else None
    value = eval_formula(formula) if formula and len(steps) == 1 else run_finqa_steps(steps)
    return None if value is None else (value, formula)


def apply_calculator(answer: str) -> str:
    """v2's answer with its final number recomputed from its own steps,
    keeping its unit word; unchanged when calculate() gives None."""
    if (result := calculate(answer)) is None:
        return answer
    value, formula = result
    head, final = answer.rsplit("Answer:", 1)
    unit = re.search(r"\b(thousand|million|billion|percent)\b", final)
    if unit and unit[1] == "percent" and not (formula and "%" in formula):
        value *= 100  # a ratio (0.393) the model reports as '39.3 percent'
    return f"{head}Answer: {value:.5f}" + (f" {unit[1]}" if unit else "")


def _earlier(results: list, step: str) -> float:
    """Result of an earlier step, 1-based ('step 0' or a later step -> IndexError)."""
    if not 1 <= int(step) <= len(results):
        raise IndexError(step)
    return results[int(step) - 1]


def run_finqa_steps(steps: list):
    """Result of the last step, or None if any step isn't plain arithmetic
    (e.g. 'take the max of ...') or doesn't parse."""
    results = []
    for step in steps:
        m = _FINQA_STEP_RE.match(step)
        if not m:
            return None
        try:
            args = [_earlier(results, ref[1]) if (ref := re.fullmatch(r"the result of step (\d+)", a)) else _number(a)
                    for a in m.group(2, 3)]
            results.append(_FINQA_OPS[m[1]](*args))
        except (ValueError, IndexError, ZeroDivisionError, OverflowError):
            return None
    return results[-1] if results else None
