"""Scoring helpers in sample_eval.py, on made-up answers. Pure Python, no GPU."""
from training.qlora_financial.sample_eval import bucket, majority_vote, pass_at_k


def test_pass_at_k_by_hand():
    assert abs(pass_at_k(16, 4, 8) - (1 - 495 / 12870)) < 1e-12  # C(12,8)=495, C(16,8)=12870
    assert pass_at_k(16, 0, 8) == 0.0
    assert pass_at_k(16, 9, 8) == 1.0  # only 7 wrong: any 8 picks include a right one
    assert abs(pass_at_k(16, 4, 1) - 4 / 16) < 1e-12  # pass@1 = plain accuracy


def test_majority_vote():
    assert majority_vote([14.1, 14.10, 3.0, None, None, None]) == 14.1  # Nones never win
    assert majority_vote([100.0, 100.5, 7.0]) == 100.0  # within 1% = same answer
    assert majority_vote([None, None]) is None
    assert majority_vote([5.0, 6.0]) == 5.0  # tie: first seen


def test_bucket():
    assert [bucket(c, 16) for c in (0, 7, 16)] == ["never", "sometimes", "always"]
