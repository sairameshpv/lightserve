"""rl_reward.py on hand-made answers in the Llama-3 chat format. No GPU."""
from training.qlora_financial.rl_reward import answer_text, reward_func, score

PROMPT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a financial analyst.<|eot_id|>"
          "<|start_header_id|>user<|end_header_id|>\n\nWhat is 151 / 211?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")


def rewards(answers, label):
    out = reward_func([PROMPT + a for a in answers], [PROMPT] * len(answers), [label] * len(answers))
    return out["rewards"].tolist()


def test_answer_text():
    assert answer_text(PROMPT + "Answer: 5<|eot_id|>") == "Answer: 5"
    assert answer_text(PROMPT + "Answer: 5") == "Answer: 5"  # cut off before the end token
    assert answer_text(PROMPT + "Answer: 5<|eot_id|><|end_of_text|>") == "Answer: 5"


def test_right_and_wrong():
    steps = "Step 1: divide 151 and 211.\nAnswer: {}<|eot_id|>"
    assert rewards([steps.format("0.71564"), steps.format("0.9")], "0.71564") == [1.0, 0.0]
    assert rewards(["0.7156<|eot_id|>"], "0.7156") == [1.0]  # plain answer, no steps


def test_one_percent_slip_scores_zero():
    assert score("Step 1: divide 151 and 211.\nAnswer: 0.7168", "0.71564") == 0.0  # within 1%, not 0.1%


def test_scale_words():
    assert score("Answer: 782 thousand", "782") == 1.0  # scale word added: forgiven
    assert score("Answer: 782 million", "782 thousand") == 0.0  # swapped: wrong


def test_no_number_and_cut_off():
    assert score("I cannot tell from the table.", "5") == 0.0
    assert score("Step 1: subtract 10 and 5.\nStep 2: divide", "5") == 0.0  # cut off before Answer:


def test_extra_logs_and_garbage():
    out = reward_func([PROMPT + "Step 1: add 2 and 3.\nAnswer: 5<|eot_id|>", PROMPT + "5<|eot_id|>"],
                      [PROMPT] * 2, ["5", "5"])
    assert out["extra_logs"]["has_steps"].tolist() == [1.0, 0.0]
    assert reward_func([None], [None], ["5"])["rewards"].tolist() == [0.0]  # bad input: 0, no crash
