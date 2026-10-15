"""Reward for the PPO / GRPO comparison (OpenRLHF `--reward.remote_url` file):
1.0 if the model's final number is right under the fair checker, else 0.0.
Setting A: no calculator -- the model must get the arithmetic right itself.

OpenRLHF calls reward_func one rollout at a time with `queries` = the whole
conversation decoded *with* special tokens, and only logs (doesn't raise)
if this function fails -- so it must never raise.
"""

import torch

from training.qlora_financial.sample_eval import match_variant

_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>"
_END_TOKENS = ("<|eot_id|>", "<|end_of_text|>")


def answer_text(query: str) -> str:
    """The model's answer: after the last assistant header, end tokens removed."""
    text = query.rsplit(_ASSISTANT, 1)[-1]
    for tok in _END_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def score(answer: str, label: str) -> float:
    """Fair-checker score; step-style answers cut off before 'Answer:' get 0
    (otherwise a step's own number could be read as the answer)."""
    if "Step " in answer and "Answer:" not in answer:
        return 0.0
    return float(match_variant(answer, label, "fair"))


def reward_func(queries, prompts, labels, **kwargs):
    rewards, has_steps = [], []
    for query, label in zip(queries, labels):
        try:
            answer = answer_text(query)
            rewards.append(score(answer, label))
            has_steps.append(float("Step " in answer))
        except Exception:  # never raise into OpenRLHF: a crash would silently drop the reward
            rewards.append(0.0)
            has_steps.append(0.0)
    r = torch.tensor(rewards)
    return {"rewards": r, "scores": r, "extra_logs": {"has_steps": torch.tensor(has_steps)}}
