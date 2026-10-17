"""rl_pairs.build_pairs on hand-made rows. No GPU, no server."""
from training.qlora_financial.rl_pairs import build_pairs

PROMPT = [{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": "151 / 211?"}]
RIGHT, SLIP, CUT = ("Step 1: divide 151 and 211.\nAnswer: 0.71564", "Step 1: divide 151 and 211.\nAnswer: 0.7168",
                    "Step 1: divide 151 and")


def row(answers, source="FinQA", group="g"):
    return {"prompt": PROMPT, "label": "Step 1: divide 151 and 211.\nAnswer: 0.71564", "source": source,
            "group": group, "answers": answers}


def test_pairs_only_from_mixed_questions():
    rows = [row([RIGHT, SLIP, RIGHT, CUT]), row([RIGHT] * 4), row([SLIP, CUT, SLIP, SLIP]), row([SLIP, RIGHT] * 2, "TAT-QA")]
    pairs, counts = build_pairs(rows)
    assert len(pairs) == 2 and [p["source"] for p in pairs] == ["FinQA", "TAT-QA"]
    assert counts == {"mixed": 2, "all_right": 1, "all_wrong": 1, "pairs_FinQA": 1, "pairs_TAT-QA": 1}
    assert counts["mixed"] + counts["all_right"] + counts["all_wrong"] == len(rows)


def test_chosen_right_rejected_wrong():
    for p in build_pairs([row([SLIP, RIGHT, CUT, SLIP], group=str(i)) for i in range(20)], seed=1)[0]:
        assert p["chosen"][0]["content"] == RIGHT and p["rejected"][0]["content"] in (SLIP, CUT)


def test_seeded():
    rows = [row([SLIP, RIGHT, CUT, RIGHT], group=str(i)) for i in range(30)]
    assert build_pairs(rows, seed=7) == build_pairs(rows, seed=7)
    assert build_pairs(rows, seed=7)[0] != build_pairs(rows, seed=8)[0]  # rejected picks differ


def test_openrlhf_dpo_format():
    """What openrlhf/datasets/reward_dataset.py preprocess_data reads with
    --data.prompt_key prompt --data.apply_chat_template: prompt + chosen must be a message list."""
    p = build_pairs([row([RIGHT, SLIP])])[0][0]
    assert set(p) == {"prompt", "chosen", "rejected", "source", "group"}
    for key in ("chosen", "rejected"):
        msgs = p["prompt"] + p[key]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
